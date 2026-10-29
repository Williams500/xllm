// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "t5_split_qkv_append_cache/framework/cache_policy.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace xllm::kernel::npu {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
t5_split_qkv_append_cache(const torch::Tensor& qkv,
                          const torch::Tensor& past_key,
                          const torch::Tensor& past_value,
                          int64_t head_num,
                          int64_t head_dim) {
  CHECK_EQ(qkv.device().type(), torch::kPrivateUse1);
  CHECK(qkv.scalar_type() == torch::kFloat16 ||
        qkv.scalar_type() == torch::kBFloat16);
  CHECK_EQ(qkv.dim(), 3);
  CHECK_EQ(past_key.dim(), 4);
  CHECK_EQ(past_key.sizes(), past_value.sizes());
  for (const auto& input : {qkv, past_key, past_value}) {
    CHECK_EQ(input.device(), qkv.device());
    CHECK_EQ(input.scalar_type(), qkv.scalar_type());
    CHECK(input.is_contiguous());
  }
  const int64_t batch = past_key.size(0);
  const int64_t length = past_key.size(2);
  CHECK_GE(batch, 1);
  CHECK_LE(batch, 64);
  CHECK_GE(head_num, 1);
  CHECK_LE(head_num, 128);
  CHECK_GE(head_dim, 1);
  CHECK_LE(head_dim, 256);
  CHECK_GE(length, 0);
  CHECK_LE(length, 8192);
  CHECK_EQ(past_key.size(1), head_num);
  CHECK_EQ(past_key.size(3), head_dim);
  CHECK_EQ(qkv.size(0), batch);
  CHECK_EQ(qkv.size(1), 1);
  CHECK_EQ(qkv.size(2), 3 * head_num * head_dim);
  CHECK_LE(batch * head_num * head_dim * (2 * length + 3), 1LL << 28);

  const c10_npu::NPUGuard guard(qkv.device());
  const auto stream = c10_npu::getCurrentNPUStream(qkv.device().index());
  for (const auto& input : {qkv, past_key, past_value}) {
    c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                               stream);
  }
  torch::Tensor query =
      torch::empty({batch, head_num, 1, head_dim}, qkv.options());
  torch::Tensor key =
      torch::empty({batch, head_num, length + 1, head_dim}, qkv.options());
  torch::Tensor value = torch::empty_like(key);
  const int64_t cache_mode = static_cast<int64_t>(
      t5_qkv::select_cache_policy(past_key, past_value, stream.stream()));
  EXEC_NPU_CMD(aclnnT5SplitQkvAppendCache,
               qkv,
               past_key,
               past_value,
               head_num,
               head_dim,
               cache_mode,
               query,
               key,
               value);
  return {query, key, value};
}
}  // namespace xllm::kernel::npu
