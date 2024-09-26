import torch
import utils
from kvcache_model import KVCacheModel


@torch.no_grad()
def autoregressive_sample(
        input_ids,
        model,
        max_length=100,
        top_k=0,
        top_p=0.0,
        temperature=1.0,
):
    model = KVCacheModel(model, top_k, top_p, temperature)
    output_ids = model.generate(input_ids, max_length)
    return output_ids


@torch.no_grad()
def speculative_sampling(
        input_ids,
        approx_model,
        target_model,
        max_length=100,
        gamma=4,
        top_k=0,
        top_p=0.0,
        temperature=1.0,
        random_seed=None,
):
    approx_model = KVCacheModel(approx_model, top_k, top_p, temperature)
    target_model = KVCacheModel(target_model, top_k, top_p, temperature)

    seq_len = input_ids.shape[ -1 ]
    T = seq_len + max_length

    while input_ids.shape[ -1 ] < T:
        prefix_len = input_ids.shape[ -1 ]

        x = approx_model.generate(input_ids, gamma)
        _ = target_model.generate(x, 1)

        # 小模型实际推到的位置
        n = prefix_len + gamma - 1

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device=input_ids.device)
            # 小模型推测第 n_seq_len + i 个 token 是 j
            j = x[ :, prefix_len + i ]
            # 如果置信度过低, 拒绝
            p = (
                    target_model.past_probs[ :, prefix_len + i - 1, j ]
                    / approx_model.past_probs[ :, prefix_len + i - 1, j ]
            )
            if r > p:
                n = prefix_len + i - 1
                break

        # 更新 input_ids, 回滚 kv-cache 和 past_probs
        input_ids = x[ :, : n + 1 ]
        approx_model.rollback(n + 1)
        target_model.rollback(n + 1)
        target_sample_token = utils.sample(
            utils.max_fn(target_model.past_probs[ :, -1, : ])
        )
        input_ids = torch.cat([ input_ids, target_sample_token ], dim=1)
    return input_ids
