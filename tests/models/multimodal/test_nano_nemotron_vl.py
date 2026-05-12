# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.nano_nemotron_vl import NemotronH_Nano_VL_V2
from vllm.transformers_utils.processors.nano_nemotron_vl import (
    NanoNemotronVLProcessor,
    _compute_aspect_preserving_size,
)


class _TextOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 0


class _ImageOnlyMultiModalConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        return 1 if modality == "image" else 0


class _ModelConfig:
    multimodal_config = _TextOnlyMultiModalConfig()


class _ImageOnlyModelConfig:
    multimodal_config = _ImageOnlyMultiModalConfig()


class _LanguageModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _MissingMultiModalModule:
    def named_parameters(self):
        raise AssertionError("multimodal weights should not be inspected")

    def load_weights(self, weights):
        raise AssertionError("multimodal weights should not be loaded")


class _AdapterModule:
    def named_parameters(self):
        return []


class _VisionModel:
    def __init__(self) -> None:
        self.loaded_weights: list[tuple[str, object]] = []

    def load_weights(self, weights):
        self.loaded_weights = list(weights)


class _EmptySeparatorTokenizer:
    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        return_attention_mask=False,
    ):
        assert add_special_tokens is False
        assert return_attention_mask is False
        return {"input_ids": [[] for _ in texts]}


@pytest.mark.parametrize(
    ("orig_w", "orig_h", "expected_size"),
    [
        (480, 320, (640, 416)),
        (640, 424, (640, 416)),
        (1152, 720, (640, 416)),
        (1280, 720, (672, 384)),
    ],
)
def test_nano_nemotron_vl_video_target_size_matches_policy_processor(
    orig_w, orig_h, expected_size
):
    assert (
        _compute_aspect_preserving_size(
            orig_w=orig_w,
            orig_h=orig_h,
            target_num_patches=1024,
            patch_size=16,
            downsample_ratio=0.5,
        )
        == expected_size
    )


def test_nano_nemotron_vl_native_video_replaces_context_tokens_only(monkeypatch):
    monkeypatch.setenv("NRL_VLLM_VIDEO_FRAME_SEPARATORS", "0")

    repl = NanoNemotronVLProcessor.get_video_repl(
        tokens_per_frame=[2, 1],
        frames_indices=[0, 1, 2, 3],
        frame_duration_ms=500,
        tokenizer=_EmptySeparatorTokenizer(),
        img_start_token_ids=[101],
        img_end_token_ids=[102],
        img_context_token_ids=[103],
        video_temporal_patch_size=2,
    )

    assert repl.full == [101, 103, 103, 102, 101, 103, 102]
    assert repl.is_embed is not None
    assert repl.is_embed(None, repl.full).tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        False,
    ]


def test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    object.__setattr__(model, "model_config", _ModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", _MissingMultiModalModule())
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("mlp1.0.weight", object()),
            ("vision_model.radio_model.encoder.weight", object()),
            ("sound_encoder.encoder.weight", object()),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]


def test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    language_weight = object()
    vision_weight = object()
    model.load_weights(
        [
            ("language_model.layers.0.weight", language_weight),
            ("vision_model.radio_model.encoder.weight", vision_weight),
        ]
    )

    assert language_model.loaded_weights == [("layers.0.weight", language_weight)]
    assert vision_model.loaded_weights == [
        ("radio_model.encoder.weight", vision_weight)
    ]


def test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights():
    model = object.__new__(NemotronH_Nano_VL_V2)
    language_model = _LanguageModel()
    vision_model = _VisionModel()
    object.__setattr__(model, "model_config", _ImageOnlyModelConfig())
    object.__setattr__(model, "language_model", language_model)
    object.__setattr__(model, "mlp1", _AdapterModule())
    object.__setattr__(model, "vision_model", vision_model)
    object.__setattr__(model, "sound_encoder", None)

    with pytest.raises(AssertionError):
        model.load_weights([("sound_encoder.encoder.weight", object())])
