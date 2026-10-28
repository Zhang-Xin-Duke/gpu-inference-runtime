#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

namespace {
// Independent implementation: segments describe request/feature rectangles.
// Adjacent lanes copy adjacent feature columns; shared rows use a zero stride.
__global__ void expand_segments(const float* source, const int64_t* descriptors,
                                float* output, int64_t width, int tiles) {
  const int64_t segment = blockIdx.x / tiles;
  const int tile = blockIdx.x % tiles;
  const int64_t* d = descriptors + segment * 6;
  const int64_t dim = d[4];
  const int64_t count = d[2] * dim;
  for (int64_t k = int64_t(tile) * blockDim.x + threadIdx.x; k < count;
       k += int64_t(tiles) * blockDim.x) {
    const int64_t row = k / dim;
    const int64_t col = k % dim;
    output[(d[1] + row) * width + d[3] + col] =
        source[d[0] + (d[5] ? 0 : row * dim) + col];
  }
}
}  // namespace

void pack_features_cuda(at::Tensor source, at::Tensor descriptors,
                        at::Tensor output, int64_t max_elements) {
  TORCH_CHECK(source.is_cuda() && descriptors.is_cuda() && output.is_cuda(),
              "packing tensors must be CUDA tensors");
  TORCH_CHECK(source.device() == output.device() && source.device() == descriptors.device(),
              "packing tensors must use the same device");
  TORCH_CHECK(source.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat &&
              descriptors.scalar_type() == at::kLong, "invalid packing dtypes");
  TORCH_CHECK(source.dim() == 1 && descriptors.dim() == 2 && descriptors.size(1) == 6 &&
              output.dim() == 2 && source.is_contiguous() && descriptors.is_contiguous() &&
              output.is_contiguous(), "invalid packing shapes/strides");
  TORCH_CHECK(descriptors.size(0) > 0 && output.numel() > 0 && max_elements > 0,
              "empty feature batch");
  const int tiles = std::min<int64_t>((max_elements + 255) / 256, 32);
  TORCH_CHECK(descriptors.size(0) <= 2147483647LL / tiles, "too many segments");
  c10::cuda::CUDAGuard guard(source.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(source.get_device());
  expand_segments<<<descriptors.size(0) * tiles, 256, 0, stream.stream()>>>(
      source.data_ptr<float>(), descriptors.data_ptr<int64_t>(),
      output.data_ptr<float>(), output.size(1), tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
