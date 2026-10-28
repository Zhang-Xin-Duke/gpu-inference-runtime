#include <torch/extension.h>

void pack_features_cuda(torch::Tensor source, torch::Tensor descriptors,
                        torch::Tensor output, int64_t max_elements);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack", &pack_features_cuda, "Fused sample copy and shared-feature expansion");
}
