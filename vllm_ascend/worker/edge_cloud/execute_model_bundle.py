# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-dp_rank state bundle for the batched compute path.

The shared-model edge worker batches head / tail model_forward across
dp_ranks: a single ``_model_forward`` + ``compute_logits`` call replaces
``dp_size`` per-dp_rank calls. The state that has to flow through the
batched forward is captured in :class:`_ExecuteModelBundle`, produced by
``NPUModelRunner.execute_model_pre`` (per-dp_rank) and consumed by
``NPUModelRunner.execute_model_batched_head`` / ``_tail`` /
``_post_batched``.

Field conventions
-----------------

* Tensors carrying block-id-derived / slot-derived indices
  (``attn_metadata.block_table_tensor``, ``attn_metadata.slot_mapping``)
  are stored **per-dp_rank with their local block_id / slot value** —
  the global offset is applied at the batched-merge step in
  :meth:`SharedModelEdgeWorker._get_or_build_merged_attn_ctx`, not at
  pre time. This keeps ``execute_model_pre`` independent of
  ``_per_dp_offsets`` (which is only finalised once all dp_ranks have
  registered their ``KVCacheConfig`` in
  ``initialize_kv_cache``).

* ``num_reqs`` / ``query_start_loc`` / ``seq_lens`` are pure per-dp_rank
  token-count values: at merge time they are concatenated with cumsum
  token offsets (no block-id offset).

* ``hidden_states`` is the per-dp_rank ``IntermediateTensors["hidden_states"]``
  coming back from the cloud (tail segment) — needed by the per-dp_rank
  post path to populate ``ExecuteModelState.hidden_states`` for
  downstream consumers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


@dataclass
class _ExecuteModelBundle:
    """Per-dp_rank state produced by ``execute_model_pre`` and consumed by
    the batched head / tail / per-dp_rank post.

    Field mapping against ``ExecuteModelState`` (set by
    ``execute_model_post_batched``) is documented in the design doc; see
    ``worker-dp-rank-worker-dp-rank-worker-mo-dapper-truffle.md``.
    """

    # Tensors to be concat'd for the batched forward. The first
    # three are ``| None`` because ``NPUModelRunner._preprocess`` may
    # legitimately return ``(None, None, ...)`` (e.g. encoder-only
    # / mm-input paths that feed the model via ``inputs_embeds``
    # only). The batched head / tail cat guards against None
    # explicitly and falls back to a merged ``None`` rather than
    # crashing on ``torch.cat``. ``inputs_embeds`` is mutually
    # exclusive with ``input_ids`` (mm path → ``inputs_embeds``
    # only, text path → ``input_ids`` only); the model runner
    # picks one of them inside ``forward_edge_cloud_segment``.
    input_ids: torch.Tensor | None
    positions: torch.Tensor | None
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    hidden_states: torch.Tensor | None

    # Per-dp_rank state for the post-batched / sample path
    logits_indices: torch.Tensor
    spec_decode_metadata: "SpecDecodeMetadata | None"
    spec_decode_common_attn_metadata: Any | None
    scheduler_output: "SchedulerOutput"

    # Per-dp_rank attention metadata — the *output* of
    # ``_build_attention_metadata`` for this dp_rank (an
    # ``AscendMetadata`` / ``GDNAttentionMetadata`` dataclass per
    # layer). The merged-attn ctx builder does NOT read derived
    # fields from this object (``num_prefills`` / ``num_decodes`` /
    # ``attn_state`` / ``actual_seq_lengths_q`` / ``attn_mask`` /
    # ``causal`` / ``model_runner_type`` etc.) — those are recomputed
    # in the batched merge step from the raw fields below. The
    # dataclass is kept here only for
    # ``execute_model_post_batched`` (per-dp_rank post, which writes
    # ``ExecuteModelState.attn_metadata``).
    attn_metadata: Any

    # Actual per-dp_rank request count (NOT the cudagraph-padded
    # ``num_reqs_padded`` stored in ``common_attn_metadata.num_reqs``).
    # The batched merge step uses this as the upper bound when
    # un-padding the per-req cm fields via
    # ``common_attn_metadata.unpadded(num_actual_tokens,
    # num_reqs_actual)`` so the merged tensor carries only real data
    # (the padded tail — filled with 0 / False by
    # ``_build_attention_metadata`` — would otherwise be cat'd as
    # if it were real). Sourced from
    # ``self.input_batch.num_reqs`` in ``execute_model_pre``.
    num_reqs_actual: int

    # Per-dp_rank ``AscendCommonAttentionMetadata`` (the
    # ``cm_base`` already produced by
    # ``NPUModelRunner._build_attention_metadata``). gid=0's
    # ``block_table_tensor`` / ``slot_mapping`` (and any other
    # gid-0-specific fields) live HERE — ``_save`` only writes
    # overrides to ``self.per_gid_cm`` for ``kv_cache_gid > 0``
    # (see ``_build_attention_metadata`` lines ~4543–4547), so
    # there is no duplication with ``per_gid_cm``.
    # The batched merge step shallow-copies this object, then
    # per-(kv_cache_gid, attn_gid) applies the matching
    # ``per_gid_cm[kv_cache_gid]`` before calling
    # ``builder.build(common_attn_metadata=cm)``.
    common_attn_metadata: "AscendCommonAttentionMetadata"
    # Per-kv_cache_gid override dicts. Each entry is the
    # ``self.per_gid_cm[kv_cache_gid]`` saved by
    # ``_save('encoder_seq_lens')`` /
    # ``_save('encoder_seq_lens_cpu')`` /
    # ``_save('query_start_loc')`` /
    # ``_save('query_start_loc_cpu')`` (GDN-only) /
    # ``_save('block_table_tensor')`` /
    # ``_save('slot_mapping')`` calls in
    # ``_build_attention_metadata``. Length
    # ``== len(self.kv_cache_config.kv_cache_groups)``.
    # The merge step applies the dict on top of a shallow
    # ``copy(common_attn_metadata)`` to produce the per-gid cm
    # the builder consumes.
    per_gid_cm: list[dict[str, Any]]   # len=num_kv_cache_gids
    # Per-(kv_cache_gid, attn_gid) ``(cascade_attn_prefix_len,
    # extra_attn_metadata_args)`` tuple saved by
    # ``_build_attn_group_metadata`` for batched-path
    # consumption. Keyed by ``(kv_cache_gid, attn_gid)``.
    # ``extra_attn_metadata_args`` is the FINAL state after
    # both GDN and DSA branches have populated it (the save
    # is placed AFTER all assignments).
    per_gid_extra: dict[tuple[int, int], tuple[int, dict[str, Any]]]

    # Forward-context fields
    num_tokens_padded: int
    num_tokens_across_dp: torch.Tensor | None
    cudagraph_mode: CUDAGraphMode
    batch_desc: BatchDescriptor | None
    ec_connector_output: Any | None
    cudagraph_stats: Any | None

    # Pre-stage side effects (from ``_update_states``)
    deferred_state_corrections_fn: Callable[[], None] | None


@dataclass
class _MergedAttnContext:
    """Cached merged forward-context state shared between head and
    tail segments.

    Built once per round on the leader worker by
    :meth:`NPUModelRunner._get_or_build_merged_attn_ctx` and consumed
    directly by ``execute_model_batched_head`` /
    ``execute_model_batched_tail``.

    Holds:
    - ``merged_attn_metadata``: the merged
      ``AscendCommonAttentionMetadata`` (the tail segment only
      mutates ``num_actual_tokens`` to ``merged_hidden.shape[0]``).
    - ``merged_batch_descriptor``: the merged
      ``BatchDescriptor`` (``num_tokens`` / ``num_reqs`` summed,
      ``uniform`` AND'd, ``has_lora`` / ``num_active_loras``
      merged) used as the cudagraph dispatch key for the merged
      batched forward. Without this, the batched head/tail would
      pass ``any_bundle.batch_desc`` (a single dp_rank's
      descriptor) but actually execute with
      ``num_tokens_padded_merged`` — inconsistent.
    - ``num_tokens_padded_merged``: cudagraph-aligned padded
      token count for the merged batch (NOT the per-dp_rank
      sum). Computed via ``cudagraph_dispatcher.dispatch`` on
      the merged token count; summing per-dp_rank padded
      counts would over-allocate the forward-context buffer.
    - ``merged_input_ids`` / ``merged_positions`` /
      ``merged_inputs_embeds``: per-dp_rank tensors trimmed to
      actual token count and concatenated. Built in
      ``execute_model_batched_head`` and reused by
      ``execute_model_batched_tail`` (which uses the same
      per-dp_rank batches). ``None`` for fields not present
      in the per-dp_rank bundles (mm path vs. text path).

    Cleared at end-of-round by
    :meth:`_BatchedExecuteMarker.drain_batched_round`.

    Lives in this module (next to :class:`_ExecuteModelBundle`) so
    that both ``model_runner_v1.py`` and
    ``shared_model_edge_worker.py`` can depend on it without
    forming an import cycle.
    """

    merged_attn_metadata: Any
    merged_batch_descriptor: Any
    merged_cudagraph_mode: "CUDAGraphMode"
    num_tokens_padded_merged: int
    merged_input_ids: torch.Tensor | None
    merged_positions: torch.Tensor | None
    merged_inputs_embeds: torch.Tensor | None

    # Decode-first reorder state.
    #
    # The batched merge cat's per-req fields in the order of
    # ``bundles`` (== ``batched_dp_ranks`` order). Many attention
    # builders (notably ``GDNAttentionMetadataBuilder``) call
    # ``split_decodes_and_prefills`` which assumes the batch is
    # already in decode-first order
    # (``first_prefill = argmax(is_prefill)`` gives ``num_decodes``
    # directly). Per-dp_rank ``_may_reorder_batch`` only sorts
    # INSIDE a single dp_rank; the merged batch still needs a
    # global decode-first reorder so the attention kernel's
    # decode/prefill split matches the merged layout.
    #
    # Both perm tensors are ``None`` iff the merged batch is
    # decode-only (``merged_attn_state == DecodeOnly``) and the
    # reorder is a no-op — in that case the cat-order layout
    # IS the decode-first layout, so the head/tail stages skip
    # every per-token permute / un-permute work below.
    # Otherwise ``merged_token_perm`` has shape
    # ``[merged_num_actual_tokens]`` (CPU int64) and maps a
    # cat-order token index to its reordered position; used by
    # the tail stage to rewrite ``merged_logits_indices`` so
    # that ``merged_sample_hidden_states = hidden_states[indices]``
    # still picks the right rows after the per-token reorder.
    merged_token_perm: torch.Tensor | None
    # ``None`` when ``merged_token_perm`` is ``None``;
    # otherwise shape ``[merged_num_actual_tokens]`` (CPU int64)
    # mapping a reordered token index back to its cat-order
    # index. Used by tail stage to un-permute ``merged_hidden``
    # so the worker can slice per-dp_rank with the original
    # cat-order ``token_offsets`` and feed
    # ``execute_model_post_batched`` with a tensor in the same
    # per-dp_rank local layout that ``bundle[i].logits_indices``
    # expects.
    inv_merged_token_perm: torch.Tensor | None
