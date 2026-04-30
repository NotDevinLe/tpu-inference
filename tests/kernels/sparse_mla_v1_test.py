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

import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

import tpu_inference.kernels.mla.v1.kernel as mla
import tpu_inference.kernels.sparse_mla.v1.kernel as sparse_mla
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, cdiv, get_dtype_packing)

jax.config.parse_flags_with_absl()


def _build_topk_indices(seq_lens, topk, seed):
    """Generate deterministic per-query topk_indices for tests.

    For each query in each sequence, select ``topk`` distinct kv positions from
    [0, kv_len). If kv_len < topk, pad with -1 sentinels. Causal-respecting
    (queries only select from kv positions <= their own causal cutoff).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for q_len, kv_len in seq_lens:
        for q_idx in range(q_len):
            causal_cutoff = kv_len - q_len + q_idx + 1
            valid = np.arange(causal_cutoff, dtype=np.int32)
            if valid.size <= topk:
                row = np.full(topk, -1, dtype=np.int32)
                row[:valid.size] = valid
            else:
                row = rng.permutation(valid)[:topk].astype(np.int32)
                row.sort()
            rows.append(row)
    return jnp.asarray(np.stack(rows, axis=0))


def _full_topk_indices(seq_lens, topk):
    """topk_indices that select every causal-valid kv position (sanity case)."""
    rows = []
    for q_len, kv_len in seq_lens:
        for q_idx in range(q_len):
            causal_cutoff = kv_len - q_len + q_idx + 1
            row = np.full(topk, -1, dtype=np.int32)
            row[:causal_cutoff] = np.arange(causal_cutoff, dtype=np.int32)
            rows.append(row)
    return jnp.asarray(np.stack(rows, axis=0))


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class SparseMlaRaggedPagedAttentionRefTest(jtu.JaxTestCase):
    """L0 tests: pure-JAX reference vs dense MLA on CPU.

    These run without TPU. They validate the masking logic of the reference
    impl against the dense MLA reference, which is the fixed point we need
    before porting to Pallas (L1).
    """

    def _setup_inputs(self, seq_lens, num_heads, lkv_dim, r_dim, page_size,
                      q_dtype, kv_dtype, num_pages):
        rng = np.random.default_rng(1234)

        def gen_random(shape, dtype):
            return jnp.array(rng.random(size=shape,
                                        dtype=np.float32)).astype(dtype)

        padded_r_dim = align_to(r_dim, 128)
        padded_lkv_dim = align_to(lkv_dim, 128)
        padded_kv_dim = padded_lkv_dim + padded_r_dim
        packing = get_dtype_packing(kv_dtype)
        q_lens = [s[0] for s in seq_lens]
        kv_lens_list = [s[1] for s in seq_lens]
        total_q_len = sum(q_lens)
        cu_q_lens_list = [0]
        for q_len in q_lens:
            cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)

        max_kv_len = max(kv_lens_list) if kv_lens_list else 0
        pages_per_seq = cdiv(max_kv_len, page_size)

        page_indices_list = []
        page_count = 0
        for kv_len in kv_lens_list:
            num_seq_pages = cdiv(kv_len, page_size)
            indices = list(range(page_count, page_count + num_seq_pages))
            page_indices_list.extend(indices + [-1] *
                                     (pages_per_seq - num_seq_pages))
            page_count += num_seq_pages

        total_num_pages = max(num_pages, page_count)

        ql_nope = gen_random((total_q_len, num_heads, lkv_dim), q_dtype)
        q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
        new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
        new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)

        cache_kv = gen_random(
            (total_num_pages, page_size // packing, packing, padded_kv_dim),
            kv_dtype,
        )
        kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
        page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
        cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)
        distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)

        return dict(
            ql_nope=ql_nope,
            q_pe=q_pe,
            new_kv_c=new_kv_c,
            new_k_pe=new_k_pe,
            cache_kv=cache_kv,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            total_q_len=total_q_len,
        )

    def test_topk_full_equals_dense_mla(self):
        """Sanity fixed point: topk == kv_len matches dense MLA exactly.

        If this fails, the topk masking in the reference is wrong. Cheap, high
        diagnostic value.
        """
        seq_lens = [(64, 64)]
        num_heads = 8
        lkv_dim = 128
        r_dim = 64
        page_size = 64
        dtype = jnp.bfloat16

        ins = self._setup_inputs(seq_lens,
                                 num_heads,
                                 lkv_dim,
                                 r_dim,
                                 page_size,
                                 dtype,
                                 dtype,
                                 num_pages=64)
        topk = max(s[1] for s in seq_lens)
        topk_indices = _full_topk_indices(seq_lens, topk)

        sparse_out, sparse_cache = (
            sparse_mla.ref_sparse_mla_ragged_paged_attention(
                ins["ql_nope"],
                ins["q_pe"],
                ins["new_kv_c"],
                ins["new_k_pe"],
                ins["cache_kv"].copy(),
                ins["kv_lens"],
                topk_indices,
                ins["page_indices"],
                ins["cu_q_lens"],
                ins["distribution"],
            ))
        dense_out, dense_cache = mla.ref_mla_ragged_paged_attention(
            ins["ql_nope"],
            ins["q_pe"],
            ins["new_kv_c"],
            ins["new_k_pe"],
            ins["cache_kv"].copy(),
            ins["kv_lens"],
            ins["page_indices"],
            ins["cu_q_lens"],
            ins["distribution"],
        )

        self.assertAllClose(sparse_out, dense_out, rtol=1e-2, atol=1e-3)
        self.assertAllClose(sparse_cache, dense_cache, rtol=1e-2, atol=1e-3)

    @parameterized.named_parameters(
        dict(testcase_name="prefill_small",
             seq_lens=[(64, 64)],
             num_heads=8,
             topk=32),
        dict(testcase_name="prefill_uneven",
             seq_lens=[(48, 200), (96, 96)],
             num_heads=8,
             topk=64),
        dict(testcase_name="decode_only",
             seq_lens=[(1, 256), (1, 1024)],
             num_heads=16,
             topk=128),
        dict(testcase_name="mixed_batch",
             seq_lens=[(64, 64), (1, 512), (32, 800)],
             num_heads=16,
             topk=64),
    )
    def test_sparse_topk_matches_self(self, seq_lens, num_heads, topk):
        """Determinism + non-NaN check on the reference for sparse topk cases.

        Once the Pallas kernel exists (L1), this test will compare reference vs
        kernel; for now it asserts the reference produces non-NaN, finite output
        and is deterministic across runs given a fixed seed.
        """
        lkv_dim = 128
        r_dim = 64
        page_size = 64
        dtype = jnp.bfloat16

        ins = self._setup_inputs(seq_lens,
                                 num_heads,
                                 lkv_dim,
                                 r_dim,
                                 page_size,
                                 dtype,
                                 dtype,
                                 num_pages=128)
        topk_indices = _build_topk_indices(seq_lens, topk, seed=42)

        out_a, _ = sparse_mla.ref_sparse_mla_ragged_paged_attention(
            ins["ql_nope"], ins["q_pe"], ins["new_kv_c"], ins["new_k_pe"],
            ins["cache_kv"].copy(), ins["kv_lens"], topk_indices,
            ins["page_indices"], ins["cu_q_lens"], ins["distribution"])
        out_b, _ = sparse_mla.ref_sparse_mla_ragged_paged_attention(
            ins["ql_nope"], ins["q_pe"], ins["new_kv_c"], ins["new_k_pe"],
            ins["cache_kv"].copy(), ins["kv_lens"], topk_indices,
            ins["page_indices"], ins["cu_q_lens"], ins["distribution"])

        self.assertEqual(
            out_a.shape,
            (ins["total_q_len"], num_heads, align_to(lkv_dim, 128)))
        self.assertTrue(jnp.all(jnp.isfinite(out_a)))
        self.assertAllClose(out_a, out_b, rtol=0.0, atol=0.0)

    def test_sentinel_minus_one_is_masked_out(self):
        """A query with all -1 sentinels has no keys; output rows for that
        query should be the row a softmax-over-all-mask produces (uniform-ish
        but with the kernel's mask_value semantics). At minimum, finite and
        not equal to the dense answer.
        """
        seq_lens = [(8, 64)]
        num_heads = 4
        lkv_dim = 128
        r_dim = 64
        page_size = 64
        dtype = jnp.bfloat16

        ins = self._setup_inputs(seq_lens,
                                 num_heads,
                                 lkv_dim,
                                 r_dim,
                                 page_size,
                                 dtype,
                                 dtype,
                                 num_pages=8)
        topk = 32
        # Mark query 0 as all-masked.
        topk_indices = _build_topk_indices(seq_lens, topk, seed=7)
        topk_indices = topk_indices.at[0].set(-1)

        out, _ = sparse_mla.ref_sparse_mla_ragged_paged_attention(
            ins["ql_nope"], ins["q_pe"], ins["new_kv_c"], ins["new_k_pe"],
            ins["cache_kv"].copy(), ins["kv_lens"], topk_indices,
            ins["page_indices"], ins["cu_q_lens"], ins["distribution"])

        self.assertTrue(jnp.all(jnp.isfinite(out)))


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class SparseMlaRaggedPagedAttentionKernelTest(jtu.JaxTestCase):
    """L1 tests: Pallas kernel vs JAX reference on TPU.

    Skips on non-TPU. Asserts that the Pallas kernel produces output
    `allclose` to the reference impl across the full test case matrix.
    Also includes the load-bearing fixed point: topk_full equals dense MLA.
    """

    def _setup_inputs(self, seq_lens, num_heads, lkv_dim, r_dim, page_size,
                      q_dtype, kv_dtype, num_pages):
        # Identical to SparseMlaRaggedPagedAttentionRefTest._setup_inputs;
        # duplicated here to keep the test classes independent.
        rng = np.random.default_rng(1234)

        def gen_random(shape, dtype):
            return jnp.array(rng.random(size=shape,
                                        dtype=np.float32)).astype(dtype)

        padded_r_dim = align_to(r_dim, 128)
        padded_lkv_dim = align_to(lkv_dim, 128)
        padded_kv_dim = padded_lkv_dim + padded_r_dim
        packing = get_dtype_packing(kv_dtype)
        q_lens = [s[0] for s in seq_lens]
        kv_lens_list = [s[1] for s in seq_lens]
        total_q_len = sum(q_lens)
        cu_q_lens_list = [0]
        for q_len in q_lens:
            cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)
        max_kv_len = max(kv_lens_list) if kv_lens_list else 0
        pages_per_seq = cdiv(max_kv_len, page_size)
        page_indices_list = []
        page_count = 0
        for kv_len in kv_lens_list:
            num_seq_pages = cdiv(kv_len, page_size)
            indices = list(range(page_count, page_count + num_seq_pages))
            page_indices_list.extend(indices + [-1] *
                                     (pages_per_seq - num_seq_pages))
            page_count += num_seq_pages
        total_num_pages = max(num_pages, page_count)
        ql_nope = gen_random((total_q_len, num_heads, lkv_dim), q_dtype)
        q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
        new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
        new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)
        cache_kv = gen_random(
            (total_num_pages, page_size // packing, packing, padded_kv_dim),
            kv_dtype,
        )
        kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
        page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
        cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)
        distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)
        return dict(ql_nope=ql_nope,
                    q_pe=q_pe,
                    new_kv_c=new_kv_c,
                    new_k_pe=new_k_pe,
                    cache_kv=cache_kv,
                    kv_lens=kv_lens,
                    page_indices=page_indices,
                    cu_q_lens=cu_q_lens,
                    distribution=distribution,
                    total_q_len=total_q_len)

    @parameterized.named_parameters(
        dict(testcase_name="prefill_small",
             seq_lens=[(64, 64)],
             num_heads=8,
             topk=32),
        dict(testcase_name="prefill_uneven",
             seq_lens=[(48, 200), (96, 96)],
             num_heads=8,
             topk=64),
        dict(testcase_name="decode_only",
             seq_lens=[(1, 256), (1, 1024)],
             num_heads=16,
             topk=128),
        dict(testcase_name="mixed_batch",
             seq_lens=[(64, 64), (2, 512), (32, 800)],
             num_heads=16,
             topk=64),
    )
    def test_kernel_matches_reference(self, seq_lens, num_heads, topk):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Sparse MLA Pallas kernel requires TPU v4+")
        lkv_dim = 128
        r_dim = 64
        page_size = 64
        dtype = jnp.bfloat16

        ins = self._setup_inputs(seq_lens,
                                 num_heads,
                                 lkv_dim,
                                 r_dim,
                                 page_size,
                                 dtype,
                                 dtype,
                                 num_pages=128)
        topk_indices = _build_topk_indices(seq_lens, topk, seed=42)

        ref_out, ref_cache = sparse_mla.ref_sparse_mla_ragged_paged_attention(
            ins["ql_nope"], ins["q_pe"], ins["new_kv_c"], ins["new_k_pe"],
            ins["cache_kv"].copy(), ins["kv_lens"], topk_indices,
            ins["page_indices"], ins["cu_q_lens"], ins["distribution"])

        kernel_out, kernel_cache = (
            sparse_mla.sparse_mla_ragged_paged_attention(
                ins["ql_nope"],
                ins["q_pe"],
                ins["new_kv_c"],
                ins["new_k_pe"],
                ins["cache_kv"].copy(),
                ins["kv_lens"],
                topk_indices,
                ins["page_indices"],
                ins["cu_q_lens"],
                ins["distribution"],
                num_kv_pages_per_block=8,
                num_queries_per_block=8,
                vmem_limit_bytes=64 * 1024 * 1024,
            ))

        self.assertAllClose(kernel_out, ref_out, rtol=1e-2, atol=1e-3)
        self.assertAllClose(kernel_cache, ref_cache, rtol=1e-2, atol=1e-3)

    def test_kernel_topk_full_equals_dense_mla(self):
        """Sanity fixed point: the Pallas kernel with topk == kv_len equals
        the dense MLA Pallas kernel. Proves the F1 mask injection is correct."""
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Sparse MLA Pallas kernel requires TPU v4+")

        seq_lens = [(64, 64)]
        num_heads = 8
        lkv_dim = 128
        r_dim = 64
        page_size = 64
        dtype = jnp.bfloat16

        ins = self._setup_inputs(seq_lens,
                                 num_heads,
                                 lkv_dim,
                                 r_dim,
                                 page_size,
                                 dtype,
                                 dtype,
                                 num_pages=64)
        topk = max(s[1] for s in seq_lens)
        topk_indices = _full_topk_indices(seq_lens, topk)

        sparse_out, sparse_cache = (
            sparse_mla.sparse_mla_ragged_paged_attention(
                ins["ql_nope"],
                ins["q_pe"],
                ins["new_kv_c"],
                ins["new_k_pe"],
                ins["cache_kv"].copy(),
                ins["kv_lens"],
                topk_indices,
                ins["page_indices"],
                ins["cu_q_lens"],
                ins["distribution"],
                num_kv_pages_per_block=8,
                num_queries_per_block=8,
                vmem_limit_bytes=64 * 1024 * 1024,
            ))

        dense_out, dense_cache = mla.mla_ragged_paged_attention(
            ins["ql_nope"],
            ins["q_pe"],
            ins["new_kv_c"],
            ins["new_k_pe"],
            ins["cache_kv"].copy(),
            ins["kv_lens"],
            ins["page_indices"],
            ins["cu_q_lens"],
            ins["distribution"],
            num_kv_pages_per_block=8,
            num_queries_per_block=8,
            vmem_limit_bytes=64 * 1024 * 1024,
        )

        self.assertAllClose(sparse_out, dense_out, rtol=1e-2, atol=1e-3)
        self.assertAllClose(sparse_cache, dense_cache, rtol=1e-2, atol=1e-3)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class SparseMlaRaggedPagedAttentionPerfTest(jtu.JaxTestCase):
    """Microbenchmark: sparse MLA Pallas kernel vs dense MLA Pallas kernel.

    Mirrors the timing pattern from ``tests/kernels/gather_reduce_test.py``:
    warmup, then 5 iterations under ``jax.block_until_ready``. Skips on
    non-TPU. Mask-based v1 still loads all KV pages, so parity with dense
    is the goal — not a speedup. Truly-sparse-load (skip non-topk pages) is
    a v2 optimization.
    """

    def _setup_inputs(self, seq_lens, num_heads, lkv_dim, r_dim, page_size,
                      q_dtype, kv_dtype, num_pages):
        rng = np.random.default_rng(1234)

        def gen_random(shape, dtype):
            return jnp.array(rng.random(size=shape,
                                        dtype=np.float32)).astype(dtype)

        padded_r_dim = align_to(r_dim, 128)
        padded_lkv_dim = align_to(lkv_dim, 128)
        padded_kv_dim = padded_lkv_dim + padded_r_dim
        packing = get_dtype_packing(kv_dtype)
        q_lens = [s[0] for s in seq_lens]
        kv_lens_list = [s[1] for s in seq_lens]
        total_q_len = sum(q_lens)
        cu_q_lens_list = [0]
        for q_len in q_lens:
            cu_q_lens_list.append(cu_q_lens_list[-1] + q_len)
        max_kv_len = max(kv_lens_list) if kv_lens_list else 0
        pages_per_seq = cdiv(max_kv_len, page_size)
        page_indices_list = []
        page_count = 0
        for kv_len in kv_lens_list:
            num_seq_pages = cdiv(kv_len, page_size)
            indices = list(range(page_count, page_count + num_seq_pages))
            page_indices_list.extend(indices + [-1] *
                                     (pages_per_seq - num_seq_pages))
            page_count += num_seq_pages
        total_num_pages = max(num_pages, page_count)
        ql_nope = gen_random((total_q_len, num_heads, lkv_dim), q_dtype)
        q_pe = gen_random((total_q_len, num_heads, r_dim), q_dtype)
        new_kv_c = gen_random((total_q_len, lkv_dim), kv_dtype)
        new_k_pe = gen_random((total_q_len, r_dim), kv_dtype)
        cache_kv = gen_random(
            (total_num_pages, page_size // packing, packing, padded_kv_dim),
            kv_dtype,
        )
        kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
        page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
        cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)
        distribution = jnp.array([0, 0, len(seq_lens)], dtype=jnp.int32)
        return dict(ql_nope=ql_nope,
                    q_pe=q_pe,
                    new_kv_c=new_kv_c,
                    new_k_pe=new_k_pe,
                    cache_kv=cache_kv,
                    kv_lens=kv_lens,
                    page_indices=page_indices,
                    cu_q_lens=cu_q_lens,
                    distribution=distribution,
                    total_q_len=total_q_len)

    @parameterized.named_parameters(
        dict(testcase_name="prefill_small",
             seq_lens=[(64, 64)],
             num_heads=8,
             topk=32,
             num_pages=64),
        dict(testcase_name="prefill_uneven",
             seq_lens=[(48, 200), (96, 96)],
             num_heads=8,
             topk=64,
             num_pages=128),
        dict(testcase_name="decode_only",
             seq_lens=[(1, 256), (1, 1024)],
             num_heads=16,
             topk=128,
             num_pages=128),
        dict(testcase_name="mixed_batch",
             seq_lens=[(64, 64), (2, 512), (32, 800)],
             num_heads=16,
             topk=64,
             num_pages=128),
    )
    def test_perf(self, seq_lens, num_heads, topk, num_pages):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Sparse MLA Pallas kernel requires TPU v4+")

        lkv_dim = 128
        r_dim = 64
        page_size = 64
        dtype = jnp.bfloat16
        n_iters = 5

        ins = self._setup_inputs(seq_lens,
                                 num_heads,
                                 lkv_dim,
                                 r_dim,
                                 page_size,
                                 dtype,
                                 dtype,
                                 num_pages=num_pages)
        topk_indices = _build_topk_indices(seq_lens, topk, seed=42)

        common_kwargs = dict(
            num_kv_pages_per_block=8,
            num_queries_per_block=8,
            vmem_limit_bytes=64 * 1024 * 1024,
        )

        def run_dense():
            return mla.mla_ragged_paged_attention(
                ins["ql_nope"], ins["q_pe"], ins["new_kv_c"], ins["new_k_pe"],
                ins["cache_kv"].copy(), ins["kv_lens"], ins["page_indices"],
                ins["cu_q_lens"], ins["distribution"], **common_kwargs)

        def run_sparse():
            return sparse_mla.sparse_mla_ragged_paged_attention(
                ins["ql_nope"], ins["q_pe"], ins["new_kv_c"], ins["new_k_pe"],
                ins["cache_kv"].copy(), ins["kv_lens"], topk_indices,
                ins["page_indices"], ins["cu_q_lens"], ins["distribution"],
                **common_kwargs)

        # Warmup (JIT compile + cache).
        for _ in range(5):
            jax.block_until_ready(run_dense())
            jax.block_until_ready(run_sparse())

        timings = {}
        start = time.time()
        for _ in range(n_iters):
            jax.block_until_ready(run_dense())
        timings["dense_mla"] = (time.time() - start) / n_iters

        start = time.time()
        for _ in range(n_iters):
            jax.block_until_ready(run_sparse())
        timings["sparse_mla"] = (time.time() - start) / n_iters

        for k, v in timings.items():
            print(f"  {k}: {v * 1000:.3f} ms / iter")
        ratio = timings["sparse_mla"] / timings["dense_mla"]
        print(f"  ratio sparse/dense: {ratio:.2f}x  (mask-based v1; "
              f"~1.0x = parity with dense, expected)")


if __name__ == "__main__":
    absltest.main()
