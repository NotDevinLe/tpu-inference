# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sparse MLA Ragged Paged Attention kernel.

Computes MLA attention restricted to a per-query top-k subset of KV positions.
Caller supplies ``topk_indices: i32[num_tokens, topk]`` (token-level, with -1
sentinel for unused slots); the kernel does the rest. KV cache layout, page
table, and softmax math are identical to the dense MLA kernel in
``tpu_inference/kernels/mla/v1/kernel.py`` — this module reuses the dense
kernel's ``update_kv_cache`` and ``get_kv_cache_shape`` helpers and adds a
sparse attention path.
"""

import jax
import jax.numpy as jnp

from tpu_inference.kernels.mla.v1.kernel import (DEFAULT_MASK_VALUE,
                                                 get_kv_cache_shape,
                                                 update_kv_cache)
from tpu_inference.kernels.ragged_paged_attention.v3.util import align_to, cdiv

__all__ = [
    "DEFAULT_MASK_VALUE",
    "get_kv_cache_shape",
    "update_kv_cache",
    "ref_sparse_mla_ragged_paged_attention",
]


def ref_sparse_mla_ragged_paged_attention(
    ql_nope: jax.Array,  # [num_tokens, actual_num_q_heads, actual_lkv_dim]
    q_pe: jax.Array,  # [num_tokens, actual_num_q_heads, actual_r_dim]
    new_kv_c: jax.Array,  # [num_tokens, actual_lkv_dim]
    new_k_pe: jax.Array,  # [num_tokens, actual_r_dim]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim+r_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    topk_indices: jax.Array,  # i32[num_tokens, topk] — -1 sentinel
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """Pure-JAX reference for sparse MLA attention.

    Differs from ``ref_mla_ragged_paged_attention`` only in that the per-query
    attention is restricted to keys listed in ``topk_indices``: positions not
    listed are masked to ``mask_value`` before softmax. Equivalent in output
    to a true gather-then-attend over the top-k subset.
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    updated_cache_kv = update_kv_cache(
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )

    actual_lkv_dim = ql_nope.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    if lkv_dim != actual_lkv_dim:
        ql_nope = jnp.pad(
            ql_nope,
            ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
            constant_values=0,
        )
    actual_r_dim = q_pe.shape[-1]
    r_dim = align_to(actual_r_dim, 128)
    if actual_r_dim != r_dim:
        q_pe = jnp.pad(q_pe, ((0, 0), (0, 0), (0, r_dim - actual_r_dim)),
                       constant_values=0)

    q = jnp.concatenate([ql_nope, q_pe], axis=-1)
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = updated_cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    assert lkv_dim == ql_nope.shape[-1]
    assert r_dim == q_pe.shape[-1]
    assert lkv_dim + r_dim == updated_cache_kv.shape[-1]

    kv_c_cache = updated_cache_kv[..., :lkv_dim].reshape(
        total_num_pages, page_size, lkv_dim)
    k_pe_cache = updated_cache_kv[...,
                                  lkv_dim:].reshape(total_num_pages, page_size,
                                                    r_dim)

    outputs = []

    for i in range(distribution[-1]):
        q_start, q_end = cu_q_lens[i], cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]

        q_i = q[q_start:q_end]  # [q_len, actual_num_q_heads, lkv_dim+r_dim]

        indices_start = i * pages_per_seq
        num_pages_i = cdiv(kv_len, page_size)
        indices_end = indices_start + num_pages_i
        indices = page_indices[indices_start:indices_end]

        gathered_kv_c = kv_c_cache[indices]
        gathered_k_pe = k_pe_cache[indices]

        flat_kv_c = gathered_kv_c.reshape(-1, lkv_dim)
        flat_k_pe = gathered_k_pe.reshape(-1, r_dim)

        k_i = jnp.concatenate([flat_kv_c[:kv_len], flat_k_pe[:kv_len]],
                              axis=-1)
        v_i = flat_kv_c[:kv_len]

        # MQA attention; attn shape: [num_heads, q_len, kv_len]
        attn = jnp.einsum("qnh,kh->nqk",
                          q_i,
                          k_i,
                          preferred_element_type=jnp.float32)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale

        # Causal mask.
        q_span = kv_len - q_len + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        causal_mask = q_span < kv_span

        # Top-k selection mask: positions not listed in topk_indices are masked.
        # topk_indices_i: [q_len, topk]; -1 entries don't match any kv position.
        topk_indices_i = topk_indices[q_start:q_end]
        kv_positions = jnp.arange(kv_len, dtype=jnp.int32)
        is_selected = jnp.any(
            topk_indices_i[:, :, None] == kv_positions[None, None, :],
            axis=1,
        )  # [q_len, kv_len]
        topk_mask = jnp.logical_not(is_selected)[
            None, :, :]  # broadcast over heads

        mask = jnp.logical_or(causal_mask, topk_mask)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)
        attn = jnp.where(mask, mask_value, attn)
        attn = jax.nn.softmax(attn, axis=-1).astype(v_i.dtype)

        out_i = jnp.einsum("nqk,kl->qnl", attn, v_i).astype(q_i.dtype)
        if v_scale is not None:
            out_i *= v_scale
        outputs.append(out_i)

    return (
        jnp.concatenate(outputs, axis=0),
        updated_cache_kv,
    )


def sparse_mla_ragged_paged_attention(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Pallas-backed sparse MLA attention.

    Thin wrapper that delegates to ``mla_ragged_paged_attention`` with the
    optional ``topk_indices`` argument set, which activates the per-query
    top-k mask path inside the existing MLA kernel (F1 architecture: one
    Pallas kernel parameterized by ``is_sparse``).
    """
    from tpu_inference.kernels.mla.v1.kernel import mla_ragged_paged_attention
    return mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        topk_indices=topk_indices,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
