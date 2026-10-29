/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace xllm::kernel::npu {
std::tuple<torch::Tensor, torch::Tensor> dual_affine_layer_norm(
    const torch::Tensor& x,
    const torch::Tensor& gamma1,
    const torch::Tensor& beta1,
    const torch::Tensor& gamma2,
    const torch::Tensor& beta2,
    double epsilon) {
  CHECK_EQ(x.device().type(), torch::kPrivateUse1);
  CHECK(x.scalar_type() == torch::kFloat16 ||
        x.scalar_type() == torch::kBFloat16);
  CHECK_EQ(x.dim(), 2);
  CHECK_GE(x.size(0), 1);
  CHECK_LE(x.size(0), 4096);
  CHECK_GE(x.size(1), 1);
  CHECK_LE(x.size(1), 8192);
  CHECK(x.is_contiguous());
  CHECK(std::isfinite(epsilon));
  CHECK_GE(epsilon, 1e-7);
  CHECK_LE(epsilon, 1e-2);
  for (const auto& affine : {gamma1, beta1, gamma2, beta2}) {
    CHECK_EQ(affine.device(), x.device());
    CHECK_EQ(affine.scalar_type(), x.scalar_type());
    CHECK_EQ(affine.dim(), 1);
    CHECK_EQ(affine.size(0), x.size(1));
    CHECK(affine.is_contiguous());
  }
  const c10_npu::NPUGuard guard(x.device());
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  for (const auto& input : {x, gamma1, beta1, gamma2, beta2}) {
    c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                               stream);
  }
  torch::Tensor y1 =
      at_npu::native::OpPreparation::apply_tensor_without_format(x);
  torch::Tensor y2 =
      at_npu::native::OpPreparation::apply_tensor_without_format(x);
  EXEC_NPU_CMD(aclnnDualAffineLayerNorm,
               x,
               gamma1,
               beta1,
               gamma2,
               beta2,
               epsilon,
               y1,
               y2);
  return {y1, y2};
}
}  // namespace xllm::kernel::npu
