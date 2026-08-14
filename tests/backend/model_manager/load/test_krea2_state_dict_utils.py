"""Unit tests for the Krea-2 loader state-dict helpers.

These cover the pure key/tensor transforms that the single-file, GGUF and Qwen3-VL encoder loaders
run before ``load_state_dict`` (prefix stripping, native<->diffusers key conversion, ComfyUI
quantization folding, encoder key remapping) plus the shared ``_reject_incomplete_load`` guard that turns a
silent partial load into an actionable error. They exercise the conversion logic without needing the
real (diffusers ``Krea2Transformer2DModel`` / transformers ``Qwen3VLModel``) architectures or weights.
"""

import json
import re
from types import SimpleNamespace

import accelerate
import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.krea2 import (
    KREA2_TRANSFORMER_CONFIG,
    _convert_krea2_native_to_diffusers,
    _dequantize_comfy_quant,
    _is_native_krea2_format,
    _normalize_qwen3vl_rope_config,
    _regular_hadamard,
    _reject_incomplete_load,
    _remap_qwen3vl_singlefile_keys,
    _strip_comfyui_prefix,
    _undo_convrot,
)


class TestNormalizeQwen3vlRopeConfig:
    def test_copies_rope_parameters_when_rope_scaling_is_missing(self) -> None:
        rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0}
        text_config = SimpleNamespace(rope_parameters=rope_parameters, rope_scaling=None)
        config = SimpleNamespace(text_config=text_config)

        assert _normalize_qwen3vl_rope_config(config) is config
        assert text_config.rope_scaling == rope_parameters

    def test_preserves_existing_rope_scaling(self) -> None:
        existing = {"rope_type": "existing"}
        text_config = SimpleNamespace(rope_parameters={"rope_type": "new"}, rope_scaling=existing)
        config = SimpleNamespace(text_config=text_config)

        _normalize_qwen3vl_rope_config(config)

        assert text_config.rope_scaling is existing

    def test_accepts_config_without_a_text_config(self) -> None:
        config = SimpleNamespace()
        assert _normalize_qwen3vl_rope_config(config) is config


class TestStripComfyuiPrefix:
    @pytest.mark.parametrize("prefix", ["model.diffusion_model.", "diffusion_model."])
    def test_strips_known_prefixes(self, prefix: str) -> None:
        sd = {f"{prefix}blocks.0.weight": torch.zeros(1), f"{prefix}first.weight": torch.zeros(1)}
        out = _strip_comfyui_prefix(sd)
        assert set(out.keys()) == {"blocks.0.weight", "first.weight"}

    def test_noop_when_no_prefix(self) -> None:
        sd = {"blocks.0.weight": torch.zeros(1), "img_in.weight": torch.zeros(1)}
        out = _strip_comfyui_prefix(sd)
        assert set(out.keys()) == set(sd.keys())

    def test_only_the_first_matching_prefix_is_used(self) -> None:
        # "model.diffusion_model." is checked before "diffusion_model.", so both strip to the same tail.
        sd = {"model.diffusion_model.blocks.0.weight": torch.zeros(1)}
        out = _strip_comfyui_prefix(sd)
        assert list(out.keys()) == ["blocks.0.weight"]


class TestIsNativeKrea2Format:
    @pytest.mark.parametrize(
        "key",
        ["blocks.0.attn.wq.weight", "txtfusion.0.mlp.up.weight", "first.weight", "blocks.0.mod.lin"],
    )
    def test_true_for_native_keys(self, key: str) -> None:
        assert _is_native_krea2_format({key: torch.zeros(1)}) is True

    @pytest.mark.parametrize(
        "key",
        ["transformer_blocks.0.attn.to_q.weight", "img_in.weight", "text_fusion.0.ff.up.weight"],
    )
    def test_false_for_diffusers_keys(self, key: str) -> None:
        assert _is_native_krea2_format({key: torch.zeros(1)}) is False


class TestDequantizeComfyQuant:
    def test_folds_scale_into_weight_and_drops_scale_key(self) -> None:
        sd = {
            "layer.weight": torch.tensor([2.0, 4.0]),
            "layer.weight_scale": torch.tensor(0.5),
        }
        out = _dequantize_comfy_quant(sd, torch.bfloat16)
        assert "layer.weight_scale" not in out
        assert torch.allclose(out["layer.weight"].float(), torch.tensor([1.0, 2.0]))

    def test_result_is_stored_in_the_compute_dtype_not_float32(self) -> None:
        """The whole model must never be materialized in float32.

        The multiply runs in float32 for precision, but holding every dequantized weight there
        costs 4 bytes per parameter: Krea-2's ~12 GB fp8 checkpoint peaked at ~50 GB of RAM before
        the caller's later bf16 cast, which swaps a 32 GB machine during a cold load.
        """
        sd = {
            "layer.weight": torch.tensor([2.0, 4.0]),
            "layer.weight_scale": torch.tensor(0.5),
        }
        assert _dequantize_comfy_quant(dict(sd), torch.bfloat16)["layer.weight"].dtype is torch.bfloat16
        assert _dequantize_comfy_quant(dict(sd), torch.float16)["layer.weight"].dtype is torch.float16

    def test_dtype_is_required(self) -> None:
        """No implicit bfloat16 fallback: on a float16-only device that would cost an extra rounding step."""
        with pytest.raises(TypeError):
            _dequantize_comfy_quant({"layer.weight": torch.tensor([2.0])})  # type: ignore[call-arg]

    def test_noop_without_scale_keys(self) -> None:
        sd = {"layer.weight": torch.tensor([2.0, 4.0])}
        out = _dequantize_comfy_quant(sd, torch.bfloat16)
        assert out is sd

    def test_orphan_scale_key_is_dropped(self) -> None:
        # A scale key with no matching weight is simply removed (nothing to multiply).
        sd = {"other.weight": torch.tensor([1.0]), "layer.weight_scale": torch.tensor(0.5)}
        out = _dequantize_comfy_quant(sd, torch.bfloat16)
        assert "layer.weight_scale" not in out
        assert "other.weight" in out

    def test_per_output_channel_scale_applies_along_the_output_dim(self) -> None:
        """A 1-D per-channel scale must scale rows, not columns.

        On a square weight both broadcasts succeed, so getting the axis wrong is silent: the model
        loads with every weight scaled by the wrong channel's factor.
        """
        sd = {
            "layer.weight": torch.ones(2, 2),
            "layer.weight_scale": torch.tensor([1.0, 3.0]),
        }
        out = _dequantize_comfy_quant(sd, torch.float32)
        assert torch.equal(out["layer.weight"], torch.tensor([[1.0, 1.0], [3.0, 3.0]]))

    def test_rejects_a_scale_that_is_neither_scalar_nor_per_channel(self) -> None:
        sd = {"layer.weight": torch.ones(2, 4), "layer.weight_scale": torch.ones(3)}
        with pytest.raises(RuntimeError, match="scalar or per-output-channel"):
            _dequantize_comfy_quant(sd, torch.float32)


class TestConvRotDequantization:
    """ConvRot checkpoints store ``W_rot = W @ H^T`` and rotate activations online.

    InvokeAI has no such kernel, so the rotation must be folded back out at load time. Skipping it
    does not fail — the shapes still match — it just loads weights in the wrong basis, which renders
    as pure noise.
    """

    @staticmethod
    def _descriptor(conf: dict[str, object]) -> torch.Tensor:
        return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)

    def test_hadamard_is_a_normalized_symmetric_involution(self) -> None:
        h = _regular_hadamard(256, "cpu", torch.float32)
        assert torch.equal(h, h.T)
        assert torch.allclose(h @ h, torch.eye(256), atol=1e-5)

    @pytest.mark.parametrize("size", [3, 8, 100])
    def test_rejects_group_sizes_that_are_not_powers_of_four(self, size: int) -> None:
        with pytest.raises(ValueError, match="power of 4"):
            _regular_hadamard(size, "cpu", torch.float32)

    def test_unrotate_inverts_the_offline_rotation(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(32, 64)
        h = _regular_hadamard(16, "cpu", torch.float32)
        rotated = torch.matmul(weight.reshape(32, 4, 16), h.T).reshape(32, 64)
        assert torch.allclose(_undo_convrot(rotated, 16), weight, atol=1e-5)

    def test_convrot_weights_are_unrotated_during_dequantization(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(8, 16)
        h = _regular_hadamard(16, "cpu", torch.float32)
        rotated = torch.matmul(weight.reshape(8, 1, 16), h.T).reshape(8, 16)
        sd = {
            "layer.weight": rotated * 2.0,
            "layer.weight_scale": torch.tensor(0.5),
            "layer.comfy_quant": self._descriptor({"format": "int8_rowwise", "convrot": True, "convrot_groupsize": 16}),
        }
        out = _dequantize_comfy_quant(sd, torch.float32)
        assert torch.allclose(out["layer.weight"], weight, atol=1e-5)
        assert "layer.comfy_quant" not in out

    def test_a_descriptor_without_convrot_leaves_the_weight_alone(self) -> None:
        sd = {
            "layer.weight": torch.tensor([[2.0, 4.0]]),
            "layer.weight_scale": torch.tensor(0.5),
            "layer.comfy_quant": self._descriptor({"format": "int8_tensorwise"}),
        }
        out = _dequantize_comfy_quant(sd, torch.float32)
        assert torch.allclose(out["layer.weight"], torch.tensor([[1.0, 2.0]]))

    def test_unsupported_quantization_format_raises_instead_of_loading_garbage(self) -> None:
        """nvfp4 and friends pack sub-byte weights into uint8; ``weight * scale`` is meaningless there."""
        sd = {
            "layer.weight": torch.zeros(4, 4, dtype=torch.uint8),
            "layer.weight_scale": torch.ones(1),
            "layer.comfy_quant": self._descriptor({"format": "nvfp4"}),
        }
        with pytest.raises(RuntimeError, match="unsupported ComfyUI quantization format"):
            _dequantize_comfy_quant(sd, torch.bfloat16)

    def test_group_size_that_does_not_divide_in_features_raises(self) -> None:
        sd = {
            "layer.weight": torch.ones(4, 20),
            "layer.weight_scale": torch.ones(1),
            "layer.comfy_quant": self._descriptor({"format": "int8_rowwise", "convrot": True, "convrot_groupsize": 16}),
        }
        with pytest.raises(RuntimeError, match="does not divide in_features"):
            _dequantize_comfy_quant(sd, torch.bfloat16)

    def test_unparseable_descriptor_falls_back_to_the_plain_scaled_path(self) -> None:
        sd = {
            "layer.weight": torch.tensor([[2.0, 4.0]]),
            "layer.weight_scale": torch.tensor(0.5),
            "layer.comfy_quant": torch.tensor([0xFF, 0xFE], dtype=torch.uint8),
        }
        out = _dequantize_comfy_quant(sd, torch.float32)
        assert torch.allclose(out["layer.weight"], torch.tensor([[1.0, 2.0]]))
        assert "layer.comfy_quant" not in out


class TestConvertKrea2NativeToDiffusers:
    def test_top_level_module_renames(self) -> None:
        sd = {
            "first.weight": torch.zeros(1),
            "tmlp.0.weight": torch.zeros(1),
            "tmlp.2.weight": torch.zeros(1),
            "tproj.1.weight": torch.zeros(1),
            "txtmlp.0.scale": torch.zeros(1),
            "txtmlp.1.weight": torch.zeros(1),
            "txtmlp.3.weight": torch.zeros(1),
            "last.linear.weight": torch.zeros(1),
            "last.norm.scale": torch.zeros(1),
        }
        out = _convert_krea2_native_to_diffusers(sd)
        assert "img_in.weight" in out
        assert "time_embed.linear_1.weight" in out
        assert "time_embed.linear_2.weight" in out
        assert "time_mod_proj.weight" in out
        assert "txt_in.norm.weight" in out
        assert "txt_in.linear_1.weight" in out
        assert "txt_in.linear_2.weight" in out
        assert "final_layer.linear.weight" in out
        assert "final_layer.norm.weight" in out

    def test_within_block_renames(self) -> None:
        sd = {
            "blocks.0.attn.wq.weight": torch.zeros(1),
            "blocks.0.attn.wk.weight": torch.zeros(1),
            "blocks.0.attn.wv.weight": torch.zeros(1),
            "blocks.0.attn.wo.weight": torch.zeros(1),
            "blocks.0.attn.gate.weight": torch.zeros(1),
            "blocks.0.attn.qknorm.qnorm.scale": torch.zeros(1),
            "blocks.0.attn.qknorm.knorm.scale": torch.zeros(1),
            "blocks.0.mlp.gate.weight": torch.zeros(1),
            "blocks.0.mlp.up.weight": torch.zeros(1),
            "blocks.0.mlp.down.weight": torch.zeros(1),
            "blocks.0.prenorm.scale": torch.zeros(1),
            "blocks.0.postnorm.scale": torch.zeros(1),
            "txtfusion.1.attn.wq.weight": torch.zeros(1),
        }
        out = _convert_krea2_native_to_diffusers(sd)
        assert "transformer_blocks.0.attn.to_q.weight" in out
        assert "transformer_blocks.0.attn.to_k.weight" in out
        assert "transformer_blocks.0.attn.to_v.weight" in out
        assert "transformer_blocks.0.attn.to_out.0.weight" in out
        assert "transformer_blocks.0.attn.to_gate.weight" in out
        assert "transformer_blocks.0.attn.norm_q.weight" in out
        assert "transformer_blocks.0.attn.norm_k.weight" in out
        assert "transformer_blocks.0.ff.gate.weight" in out
        assert "transformer_blocks.0.ff.up.weight" in out
        assert "transformer_blocks.0.ff.down.weight" in out
        assert "transformer_blocks.0.norm1.weight" in out
        assert "transformer_blocks.0.norm2.weight" in out
        # text_fusion tower renamed the same way.
        assert "text_fusion.1.attn.to_q.weight" in out
        # No native names survive.
        assert not any(".wq." in k or ".qknorm." in k or ".mlp." in k or "prenorm" in k for k in out)

    def test_final_block_projections_are_dropped(self) -> None:
        sd = {"last.down.weight": torch.zeros(2, 2), "last.up.weight": torch.zeros(2, 2)}
        out = _convert_krea2_native_to_diffusers(sd)
        assert out == {}

    def test_mod_lin_is_reshaped_to_scale_shift_table(self) -> None:
        # A flat (6*H,) per-block modulation vector becomes a (6, H) scale_shift_table.
        sd = {"blocks.0.mod.lin": torch.arange(12, dtype=torch.float32)}
        out = _convert_krea2_native_to_diffusers(sd)
        assert "transformer_blocks.0.scale_shift_table" in out
        table = out["transformer_blocks.0.scale_shift_table"]
        assert table.shape == (6, 2)
        assert torch.equal(table, torch.arange(12, dtype=torch.float32).reshape(6, 2))

    def test_final_layer_modulation_is_reshaped_to_two_by_hidden(self) -> None:
        # Krea2FinalLayer.scale_shift_table is (2, hidden) (scale, shift). The flat native
        # last.modulation.lin must be reshaped (not merely renamed), otherwise load_state_dict(assign=True)
        # installs a wrong-shaped 1-D parameter that the meta-only completeness guard cannot catch and the
        # final layer fails at inference.
        sd = {"last.modulation.lin": torch.arange(8, dtype=torch.float32)}  # 2 * hidden, hidden=4
        out = _convert_krea2_native_to_diffusers(sd)
        assert "final_layer.scale_shift_table" in out
        table = out["final_layer.scale_shift_table"]
        assert table.shape == (2, 4)
        assert torch.equal(table, torch.arange(8, dtype=torch.float32).reshape(2, 4))

    def test_non_string_keys_pass_through(self) -> None:
        sentinel = object()
        out = _convert_krea2_native_to_diffusers({sentinel: torch.zeros(1)})  # type: ignore[dict-item]
        assert sentinel in out

    @pytest.mark.parametrize(
        "keys",
        [
            ("blocks.0.attn.wq.weight", "transformer_blocks.0.attn.to_q.weight"),
            ("transformer_blocks.0.attn.to_q.weight", "blocks.0.attn.wq.weight"),
        ],
    )
    def test_rejects_mixed_layout_alias_collision(self, keys: tuple[str, str]) -> None:
        # A malformed mixed-layout checkpoint carries a native key and its already-diffusers alias, both
        # normalizing to transformer_blocks.0.attn.to_q.weight. Distinct tensors under colliding aliases
        # must be rejected in either insertion order rather than silently dropping one.
        first, second = keys
        sd = {first: torch.zeros(1), second: torch.ones(1)}
        with pytest.raises(RuntimeError, match="both normalize to"):
            _convert_krea2_native_to_diffusers(sd)


class TestRemapQwen3vlSinglefileKeys:
    def test_routes_towers_and_prefixes_bare_language_model_keys(self) -> None:
        sd = {
            "model.visual.blocks.0.weight": torch.zeros(1),
            "model.language_model.layers.0.weight": torch.zeros(1),
            "model.layers.1.weight": torch.zeros(1),  # bare LM key under a model. prefix
            "model.embed_tokens.weight": torch.zeros(1),
            "model.norm.weight": torch.zeros(1),
            "visual.blocks.1.weight": torch.zeros(1),  # already un-prefixed
            "layers.2.weight": torch.zeros(1),  # bare, no model. prefix
        }
        out = _remap_qwen3vl_singlefile_keys(sd)
        assert "visual.blocks.0.weight" in out
        assert "language_model.layers.0.weight" in out
        assert "language_model.layers.1.weight" in out
        assert "language_model.embed_tokens.weight" in out
        assert "language_model.norm.weight" in out
        assert "visual.blocks.1.weight" in out
        assert "language_model.layers.2.weight" in out
        # No key retains the leading model. prefix.
        assert not any(k.startswith("model.") for k in out)

    @pytest.mark.parametrize(
        "keys",
        [
            ("model.layers.1.weight", "layers.1.weight"),
            ("layers.1.weight", "model.layers.1.weight"),
        ],
    )
    def test_rejects_mixed_layout_alias_collision(self, keys: tuple[str, str]) -> None:
        # A bare LM key and its model.-prefixed alias both route to language_model.layers.1.weight.
        # Distinct tensors under colliding aliases must be rejected in either order, not silently merged.
        first, second = keys
        sd = {first: torch.zeros(1), second: torch.ones(1)}
        with pytest.raises(RuntimeError, match="both normalize to"):
            _remap_qwen3vl_singlefile_keys(sd)


class TestRejectIncompleteLoad:
    @pytest.mark.parametrize(
        "what",
        ["Krea-2 single-file checkpoint", "Krea-2 GGUF checkpoint", "Qwen3-VL encoder checkpoint"],
    )
    def test_raises_when_parameters_remain_on_meta_device(self, what: str) -> None:
        # accelerate.init_empty_weights() leaves every parameter on the meta device — the exact state a
        # strict=False load produces for a checkpoint that omits required weights. All three Krea-2 loaders
        # feed their `what` label through this guard, so parametrize over the real call-site messages.
        with accelerate.init_empty_weights():
            model = torch.nn.Linear(4, 4)
        with pytest.raises(RuntimeError, match=re.escape(f"{what} is incomplete")):
            _reject_incomplete_load(model, what=what)

    def test_does_not_raise_for_a_fully_materialized_model(self) -> None:
        model = torch.nn.Linear(4, 4)  # normal construction — no meta tensors
        _reject_incomplete_load(model, what="Krea-2 single-file checkpoint")

    def test_names_the_missing_parameters(self) -> None:
        # Materialize only the weight; the bias stays on meta and must be named in the error.
        with accelerate.init_empty_weights():
            model = torch.nn.Linear(4, 4)
        model.load_state_dict({"weight": torch.zeros(4, 4)}, strict=False, assign=True)
        with pytest.raises(RuntimeError, match="bias") as exc_info:
            _reject_incomplete_load(model, what="Krea-2 single-file checkpoint")
        assert "1 tensor(s)" in str(exc_info.value)

    def test_raises_for_a_persistent_buffer_left_on_meta(self) -> None:
        # Buffers land on the meta device too; a native/GGUF checkpoint that omits a persistent buffer
        # must be rejected. The parameter here is fully materialized, so a parameters-only guard would
        # wrongly pass - only the buffer check catches it.
        class _ModuleWithMetaBuffer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2, 2))
                self.register_buffer("cached_stat", torch.empty(4, device="meta"))

        model = _ModuleWithMetaBuffer()
        with pytest.raises(RuntimeError, match="cached_stat"):
            _reject_incomplete_load(model, what="Krea-2 GGUF checkpoint")


def _native_key_for_scale_shift_table(diffusers_key: str) -> str:
    """Inverse of the converter's ``.mod.lin``/``last.modulation.lin`` -> ``.scale_shift_table`` mapping.

    Only the modulation tables are reshaped by ``_convert_krea2_native_to_diffusers`` (every other
    conversion is a pure, shape-preserving rename), so these are the only keys whose converted shape can
    silently disagree with the real ``Krea2Transformer2DModel`` module dims.
    """
    stem = diffusers_key[: -len(".scale_shift_table")]
    if stem == "final_layer":
        return "last.modulation.lin"
    if stem.startswith("transformer_blocks."):
        return "blocks." + stem[len("transformer_blocks.") :] + ".mod.lin"
    if stem.startswith("text_fusion."):
        return "txtfusion." + stem[len("text_fusion.") :] + ".mod.lin"
    raise AssertionError(f"unmapped scale_shift_table key: {diffusers_key}")


class TestConvertedShapesMatchRealKrea2Transformer:
    """Validate the native->diffusers converter against the REAL ``Krea2Transformer2DModel`` module.

    The loader boundary tests use a tiny stub transformer, so they cannot detect a converted tensor whose
    shape disagrees with the real module: ``load_state_dict(assign=True)`` installs whatever shape the
    converter produced (no shape check) and ``_reject_incomplete_load`` only inspects the meta device, not
    shapes, so a wrong-shaped tensor survives load and fails only at inference. The concrete instance is
    ``last.modulation.lin`` -> ``final_layer.scale_shift_table``: the real table is ``(2, hidden)`` and the
    flat native vector must be reshaped, not merely renamed.
    """

    def test_scale_shift_tables_match_real_module_dims(self) -> None:
        pytest.importorskip("diffusers")
        from diffusers import Krea2Transformer2DModel

        with accelerate.init_empty_weights():
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)

        # Meta-device state dict still carries real shapes for every parameter/buffer.
        expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        table_keys = [name for name in expected if name.endswith(".scale_shift_table")]
        assert table_keys, "real Krea2Transformer2DModel exposes no scale_shift_table params"
        assert "final_layer.scale_shift_table" in table_keys
        assert any(k.startswith("transformer_blocks.") for k in table_keys)

        # Build a native state dict covering every modulation-table site, sized (as a flat vector) from the
        # real module dims, then convert and assert the reshaped result matches the real parameter shape.
        native_sd = {}
        for name in table_keys:
            rows, hidden = expected[name]
            native_sd[_native_key_for_scale_shift_table(name)] = torch.arange(rows * hidden, dtype=torch.float32)

        out = _convert_krea2_native_to_diffusers(native_sd)

        for name in table_keys:
            assert name in out, f"converter did not produce {name}"
            assert tuple(out[name].shape) == expected[name], (
                f"{name}: converted shape {tuple(out[name].shape)} != real module shape {expected[name]}"
            )

        # The two named layouts from the review: final layer AdaLN table is (2, hidden); every per-block
        # (transformer + text-fusion) table is (6, hidden). A regression that dropped the reshape would leave
        # these 1-D and this would fail.
        assert expected["final_layer.scale_shift_table"][0] == 2
        for name in table_keys:
            if name.startswith(("transformer_blocks.", "text_fusion.")):
                assert expected[name][0] == 6, f"{name} expected 6 modulation rows, got {expected[name]}"
