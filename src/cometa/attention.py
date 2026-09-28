"""Dense reference attention for a causal prefix with independent branches.

This is a correctness backend, not a sparse/FlashAttention implementation.
Boolean masks here use True = visible, matching PyTorch SDPA (not MultiheadAttention).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def _validate(prefix_length: int, branch_lengths: Sequence[int]) -> None:
    if prefix_length < 1 or not branch_lengths or any(n < 1 for n in branch_lengths):
        raise ValueError("A tree needs a nonempty prefix and nonempty branches")


def tree_visibility(prefix_length: int, branch_lengths: Sequence[int], *, device=None) -> torch.Tensor:
    """Return [tokens, tokens], row=query and column=key."""
    _validate(prefix_length, branch_lengths)
    total = prefix_length + sum(branch_lengths)
    visible = torch.zeros((total, total), dtype=torch.bool, device=device)
    visible[:prefix_length, :prefix_length] = torch.ones(
        (prefix_length, prefix_length), dtype=torch.bool, device=device
    ).tril()
    start = prefix_length
    for length in branch_lengths:
        visible[start:start + length, :prefix_length] = True
        visible[start:start + length, start:start + length] = torch.ones(
            (length, length), dtype=torch.bool, device=device
        ).tril()
        start += length
    return visible


def tree_attention_mask(prefix_length: int, branch_lengths: Sequence[int], *, dtype=torch.float32,
                        device=None) -> torch.Tensor:
    """Additive [1, 1, tokens, tokens] mask accepted by HF eager and SDPA."""
    if not dtype.is_floating_point:
        raise TypeError("An additive attention mask needs a floating dtype")
    visible = tree_visibility(prefix_length, branch_lengths, device=device)
    mask = torch.zeros(visible.shape, dtype=dtype, device=device)
    mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return mask[None, None]


def tree_position_ids(prefix_length: int, branch_lengths: Sequence[int], *, device=None) -> torch.Tensor:
    """Each branch continues from prefix_length, regardless of its packed offset."""
    _validate(prefix_length, branch_lengths)
    positions = [torch.arange(prefix_length, device=device)]
    positions.extend(torch.arange(prefix_length, prefix_length + n, device=device) for n in branch_lengths)
    return torch.cat(positions).unsqueeze(0)


def tree_readout_positions(prefix_length: int, branch_lengths: Sequence[int]) -> list[int]:
    _validate(prefix_length, branch_lengths)
    end = prefix_length
    result = []
    for length in branch_lengths:
        end += length
        result.append(end - 1)
    return result
