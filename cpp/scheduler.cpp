#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using Ns = std::chrono::nanoseconds;
struct Request {
    uint64_t id;
    std::string key;
    size_t samples, bytes;
    Time enqueued, deadline;
};
struct Batch {
    std::vector<uint64_t> ids, expired;
    std::string key;
    size_t samples = 0;
    bool stopped = false;
};

// Metadata only. Tensor ownership and CUDA completion belong to the backend.
class Scheduler {
    std::mutex mu_;
    std::condition_variable cv_;
    size_t max_samples_, max_requests_, max_bytes_, max_inflight_;
    Ns delay_;
    std::string policy_;
    bool closing_ = false;
    uint64_t next_id_ = 1, accepted_ = 0, rejected_ = 0, expired_ = 0;
    size_t used_bytes_ = 0, peak_bytes_ = 0, active_batches_ = 0;
    std::deque<Request> queue_;
    std::unordered_map<uint64_t, Request> active_;
    std::unordered_map<std::string, double> cost_ns_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> batches_;

    void release(const Request& r) { used_bytes_ -= r.bytes; }
    double estimate(const Request& r) const {
        auto it = cost_ns_.find(r.key);
        return it == cost_ns_.end() ? 1e7 : it->second;
    }
public:
    Scheduler(size_t samples, size_t requests, size_t bytes, size_t inflight,
              double delay_ms, std::string policy)
        : max_samples_(samples), max_requests_(requests), max_bytes_(bytes),
          max_inflight_(inflight), delay_(static_cast<int64_t>(delay_ms * 1e6)),
          policy_(std::move(policy)) {
        if (!samples || !requests || !bytes || !inflight || delay_ms < 0 ||
            !std::isfinite(delay_ms) || delay_ms > 86400000 || (policy_ != "age" && policy_ != "deadline"))
            throw std::invalid_argument("invalid scheduler limits or policy");
    }
    uint64_t offer(std::string key, size_t samples, size_t bytes, double timeout_ms) {
        std::lock_guard lock(mu_);
        if (!std::isfinite(timeout_ms) || timeout_ms <= 0 || timeout_ms > 86400000)
            throw std::invalid_argument("timeout_ms must be finite, positive and at most one day");
        if (closing_ || !samples || samples > max_samples_ || !bytes ||
            queue_.size() + active_.size() >= max_requests_ || bytes > max_bytes_ ||
            used_bytes_ > max_bytes_ - bytes) {
            ++rejected_;
            throw std::runtime_error("admission rejected: closed or capacity exceeded");
        }
        auto now = Clock::now();
        auto id = next_id_++;
        queue_.push_back({id, std::move(key), samples, bytes, now,
                          now + Ns(static_cast<int64_t>(timeout_ms * 1e6))});
        used_bytes_ += bytes;
        peak_bytes_ = std::max(peak_bytes_, used_bytes_);
        ++accepted_;
        cv_.notify_all();
        return id;
    }
    Batch take() {
        std::unique_lock lock(mu_);
        for (;;) {
            Batch result;
            auto now = Clock::now();
            for (auto it = queue_.begin(); it != queue_.end();) {
                if (now >= it->deadline) {
                    result.expired.push_back(it->id);
                    release(*it);
                    it = queue_.erase(it);
                    ++expired_;
                } else ++it;
            }
            // Return expired IDs promptly even when no execution slot is free.
            if (!result.expired.empty()) return result;
            if (queue_.empty()) {
                if (closing_) { result.stopped = true; return result; }
                cv_.wait(lock);
                continue;
            }
            if (active_batches_ >= max_inflight_) {
                auto deadline = std::min_element(queue_.begin(), queue_.end(),
                    [](const auto& a, const auto& b) { return a.deadline < b.deadline; });
                cv_.wait_until(lock, deadline->deadline);
                continue;
            }
            // Each group's oldest request anchors its age. Deadline policy uses
            // the earliest deadline minus an EWMA of observed service time
            // (10 ms before the first observation; this is a heuristic, not an SLO).
            struct Group { size_t first, samples; Time flush; };
            std::unordered_map<std::string, size_t> group_index;
            std::vector<Group> groups;
            // One metadata scan builds all group readiness estimates. Retain
            // first-seen order for deterministic age ties across model versions.
            for (size_t i = 0; i < queue_.size(); ++i) {
                const auto& r = queue_[i];
                auto [it, inserted] = group_index.emplace(r.key, groups.size());
                if (inserted) groups.push_back({i, 0, r.enqueued + delay_});
                auto& group = groups[it->second];
                if (r.samples <= max_samples_ - group.samples) group.samples += r.samples;
                else group.flush = std::min(group.flush, now);
                if (policy_ == "deadline")
                    group.flush = std::min(group.flush, r.deadline - Ns(static_cast<int64_t>(estimate(r))));
                if (closing_ || group.samples == max_samples_)
                    group.flush = std::min(group.flush, now);
            }
            auto chosen_group = std::min_element(groups.begin(), groups.end(),
                [](const auto& a, const auto& b) { return a.flush < b.flush; });
            size_t chosen = chosen_group->first;
            Time chosen_flush = chosen_group->flush;
            if (chosen_flush > now) { cv_.wait_until(lock, chosen_flush); continue; }
            result.key = queue_[chosen].key;
            for (auto it = queue_.begin(); it != queue_.end();) {
                if (it->key == result.key && it->samples <= max_samples_ - result.samples) {
                    result.samples += it->samples;
                    result.ids.push_back(it->id);
                    active_.emplace(it->id, *it);
                    it = queue_.erase(it);
                } else ++it;
            }
            batches_.emplace(result.ids.front(), result.ids);
            ++active_batches_;
            return result;
        }
    }
    // 1: removed while queued; 2: executing, backend retains its reservation;
    // 0: already terminal. Cancellation never reclaims live GPU buffers.
    int cancel(uint64_t id) {
        std::lock_guard lock(mu_);
        for (auto it = queue_.begin(); it != queue_.end(); ++it) if (it->id == id) {
            release(*it); queue_.erase(it); cv_.notify_all(); return 1;
        }
        return active_.contains(id) ? 2 : 0;
    }
    void finish(const std::vector<uint64_t>& ids, double service_ms) {
        std::lock_guard lock(mu_);
        if (ids.empty() || !std::isfinite(service_ms) || service_ms < 0)
            throw std::invalid_argument("invalid completion");
        auto batch = batches_.find(ids.front());
        if (batch == batches_.end() || batch->second != ids)
            throw std::logic_error("completion must match one entire batch");
        for (auto id : ids) if (!active_.contains(id))
            throw std::logic_error("duplicate or unknown completion");
        std::string key = active_.at(ids[0]).key;
        for (auto id : ids) { release(active_.at(id)); active_.erase(id); }
        batches_.erase(batch);
        --active_batches_;
        double measured = std::max(1.0, service_ms * 1e6);
        auto it = cost_ns_.find(key);
        cost_ns_[key] = it == cost_ns_.end() ? measured : .8 * it->second + .2 * measured;
        cv_.notify_all();
    }
    void close() { std::lock_guard lock(mu_); closing_ = true; cv_.notify_all(); }
    std::unordered_map<std::string, uint64_t> stats() {
        std::lock_guard lock(mu_);
        return {{"accepted", accepted_}, {"rejected", rejected_}, {"expired", expired_},
                {"queued", queue_.size()}, {"active_requests", active_.size()},
                {"active_batches", active_batches_}, {"reserved_bytes", used_bytes_},
                {"peak_reserved_bytes", peak_bytes_}};
    }
};
PYBIND11_MODULE(_scheduler, m) {
    py::class_<Batch>(m, "Batch")
        .def_readonly("ids", &Batch::ids).def_readonly("expired", &Batch::expired)
        .def_readonly("key", &Batch::key).def_readonly("samples", &Batch::samples)
        .def_readonly("stopped", &Batch::stopped);
    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init<size_t,size_t,size_t,size_t,double,std::string>(),
             py::arg("max_samples"), py::arg("max_requests"), py::arg("max_bytes"),
             py::arg("max_inflight"), py::arg("delay_ms"), py::arg("policy") = "deadline")
        .def("offer", &Scheduler::offer).def("take", &Scheduler::take,
             py::call_guard<py::gil_scoped_release>())
        .def("cancel", &Scheduler::cancel).def("finish", &Scheduler::finish)
        .def("close", &Scheduler::close).def("stats", &Scheduler::stats);
}
