# L40 multi-GPU recordings — October 28, 2026

[Evaluation report](../../docs/MULTI_GPU_EVALUATION.md) describes the conditions,
results and remaining recovery failures. All GPU instances were deleted after
artifact collection.

## Final measurements

- [`fixed/host-0/results/compute/`](fixed/host-0/results/compute/): matched
  one/two-device compute comparison, two seeds, 10-second cells at nominal
  1,000 requests/s. The runtime and evaluation source are identified by
  [`fixed/source-manifest.json`](fixed/source-manifest.json), based on commit
  `2419b663a47ed75258adbef80df3bd280383625f`.
- [`fixed/failover.json`](fixed/failover.json): 75-second three-replica recording
  across two hosts, including the stop/restart events and all failed requests.
- [`fixed/recovery-summary.json`](fixed/recovery-summary.json): derived recovery
  and connection accounting. Rejoin timing is sampled rather than exact.
- [`fixed/failover-presentation.json`](fixed/failover-presentation.json): the
  same observations with a title and topology caption for the animation.
- `fixed/host-*-setup.log`: both GPU hosts passed 33 tests, including CUDA cases.

The final compute comparison reaches median deadline goodput of
244.10 / 519.85 requests/s for one/two devices. Both configurations are overloaded;
the raw outcomes include rejection, expiration and errors. The three-replica run
records 6,818 on-time requests and 667 failures out of 7,485 offered requests.

## Earlier measurements retained for diagnosis

- `meaningful/host-0/results/compute/`: a 12-cell compute-heavy sweep before the
  connection-pool fix, using 8-second cells at 200/600/1,000 requests/s.
- `meaningful/failover.json`: a 45-second recovery run before the fix; 1,574
  on-time requests, 2,944 failures. This differs in duration from the final run
  and is not a matched recovery-rate comparison.
- `sweep/`: initial WAN-limited small-model 1/2/3-replica sweep. Its outer
  controller timed out at the final cell and deleted the instances. The final
  `replicas-1-rps-600-seed-17.json` may include that shutdown and is excluded from
  performance conclusions. The planned recovery phase did not run.
- `connection-pool-before.log`: deterministic failing regression for excess
  idle connections in the earlier per-thread cache.

The deployment uses provider-virtualized GPU allocations. `topology.json` and
environment logs describe the allocated devices and visibility selections;
changing UUID snapshots are not used to establish physical GPU isolation.

## Rebuild the final presentation

From the repository root:

```bash
python benchmarks/summarize_replicas.py \
  --root results/three-l40-2026-10-28/fixed/host-0/results/compute \
  --title 'Compute-heavy FP32 inference: one vs two L40 devices, local HTTP'
python benchmarks/render_distributed.py \
  --input results/three-l40-2026-10-28/fixed/failover-presentation.json \
  --output results/three-l40-2026-10-28/fixed/three-replica-recovery.gif
```

`SHA256SUMS` covers this artifact directory except the checksum file itself.
Checked-in raw files preserve their bytes across platform checkouts.
