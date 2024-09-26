import torch
import utils


class KVCacheModel:
    def __init__(self, model, top_k=0, top_p=0.0, temperature=1.0):
        self.model = model
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        # kv-cache 的形状为 (n_layers, 2, bs, num_heads, seq_len, head_dim)
        self.past_key_values = None
        # past_probs 的形状为 (batch_size, seq_len, vocab_size)
        self.past_probs = None

    def _forward_with_kvcache(self, input_ids):
        # 首次调用, 初始化 past_key_values 和 past_probs
        if self.past_key_values is None:
            assert self.past_probs is None
            outputs = self.model(input_ids=input_ids)

            # 存储 norm 后的 probs
            # 形状为 (batch_size, seq_len, vocab_size)
            self.past_probs = outputs.logits
            for i in range(self.past_probs.shape[ -2 ]):
                self.past_probs[ :, i, : ] = utils.norm_logits(
                    self.past_probs[ :, i, : ], self.top_k, self.top_p, self.temperature
                )
            # 存储 key_values
            self.past_key_values = outputs.past_key_values

            last_probs = self.past_probs[ :, -1, : ]

        else:
            cache_len = 0
            for kv in self.past_key_values:
                k, _ = kv
                cache_len = k.shape[ 2 ]
                break
            last_input_ids = input_ids[ :, cache_len: ]
            if last_input_ids.dim() == 1:
                last_input_ids = last_input_ids.unsqueeze(0)

            output = self.model(
                last_input_ids, past_key_values=self.past_key_values, use_cache=True
            )

            not_cached_q = output.logits
            if not_cached_q.dim() == 2:
                not_cached_q = not_cached_q.unsqueeze(0)
            for i in range(not_cached_q.shape[ -2 ]):
                not_cached_q[ :, i, : ] = utils.norm_logits(
                    not_cached_q[ :, i, : ], self.top_k, self.top_p, self.temperature
                )
            # 按照 seq_len 维度合并 past_probs
            self.past_probs = torch.cat((self.past_probs, not_cached_q), dim=1)
            # 更新 kv-cache
            self.past_key_values = output.past_key_values

            last_probs = not_cached_q[ :, -1, : ]
        # 返回生成的 token probs
        return last_probs

    @torch.no_grad()
    def generate(self, input_ids, gamma):
        """往后推 gamma 步
        """
        x = input_ids
        for _ in range(gamma):
            probs = self._forward_with_kvcache(x)
            next_token = utils.sample(probs)
            x = torch.cat((x, next_token), dim=1)
        return x

    @torch.no_grad()
    def rollback(self, end_pos):
        """ 将 kv-cache 和 past_probs 回滚到 end_pos 之前
        """
        self.past_probs = self.past_probs[ :, :end_pos, : ]
        past_key_values_new = [ ]
        for kv in self.past_key_values:
            k, v = kv
            # 单层中, k 和 v 的形状都是 (bs, num_heads, seq_len, head_dim)
            k = k[ :, :, :end_pos, : ]
            v = v[ :, :, :end_pos, : ]
            kv_new = (k, v)
            past_key_values_new.append(kv_new)
        self.past_key_values = past_key_values_new
