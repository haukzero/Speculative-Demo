import torch
from torch.nn import functional as F


def _top_k_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """先取 top_k, 再取 top_p, 每一步将不符合的置为 -inf

    Args:
        logits (torch.Tensor): (bs, n_vocab)
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0..
    """
    if top_k > 0:
        top_k_val = torch.topk(logits, min(top_k, logits.shape[ -1 ]), dim=-1)[ 0 ]
        logits[ logits < top_k_val[ :, -1 ].unsqueeze(-1) ] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # 计算累计和
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[ :, 1: ] = filter[ :, :-1 ].clone()
        filter[ :, 0 ] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[ indices_to_remove ] = float("-inf")
    return logits


def norm_logits(
        logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, temperature: float = 1.0
):
    """将 logits 先后选取 top_k 和 top_p 后取 softmax

    Args:
        logits (torch.Tensor): (bs, n_vocab)
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0..
        temperature (float, optional): Defaults to 1..

    Returns:
        probs: (bs, n_vocab)
    """
    logits = _top_k_p_filter(logits / temperature, top_k, top_p)
    return F.softmax(logits, dim=-1)


def sample(probs: torch.Tensor, n_smp=1):
    return torch.multinomial(probs, n_smp)


def max_fn(x: torch.Tensor):
    re = F.relu(x)
    return re / re.sum(dim=-1, keepdim=True)


def blue_print(s):
    print(f"\033[1;34m{s}\033[0m")


def green_print(s):
    print(f"\033[1;32m{s}\033[0m")
