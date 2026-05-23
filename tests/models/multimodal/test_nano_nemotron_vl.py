# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models import nano_nemotron_vl as nano_module
from vllm.model_executor.models.nano_nemotron_vl import (
    NanoNemotronVLMultiModalProcessor,
    NemotronH_Nano_VL_V2,
)
from vllm.multimodal.evs import compute_placeholder_tokens_per_frame
from vllm.multimodal.processing.processor import PromptUpdateDetails
from vllm.transformers_utils.processors.nano_nemotron_vl import (
    AUDIO_CONTEXT,
    BaseNanoNemotronVLProcessor,
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


class _EmbeddingLanguageModel:
    def embed_input_ids(self, token_ids):
        token_ids = token_ids.float()
        return torch.stack((token_ids, -token_ids), dim=-1)


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


class _SeparatorTokenizer:
    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        return_attention_mask=False,
    ):
        assert add_special_tokens is False
        assert return_attention_mask is False
        return {"input_ids": [[200 + idx] for idx, _ in enumerate(texts)]}


class _AudioContextTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        if text == AUDIO_CONTEXT:
            return [901, 902]
        return [1]


class _PromptInfo:
    supports_audio = True

    def get_tokenizer(self):
        return _AudioContextTokenizer()


class _TokenizedText:
    input_ids = [1, 2, 3]


class _ImageTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return _TokenizedText()


class _FakeDynamicTiler:
    def __init__(self) -> None:
        self.called = False
        self._max_num_patches = 4096

    def _images_to_pixel_values_lst(self, *, text_prompt_length, images, dtype):
        self.called = True
        assert text_prompt_length == len(_TokenizedText.input_ids)
        assert len(images) == 1
        assert dtype is torch.float32
        return [torch.zeros(3, 16, 16)], [7]


class _DummyImageProcessor(BaseNanoNemotronVLProcessor):
    @property
    def image_token_id(self) -> int:
        return 0

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: int | None,
    ) -> PromptUpdateDetails[str]:
        return PromptUpdateDetails.from_seq(f"<repl:{feature_size}:{num_patches}>")


def _new_dummy_image_processor():
    processor = object.__new__(_DummyImageProcessor)
    processor.tokenizer = _ImageTokenizer()
    processor.dtype = torch.float32
    processor.dynamic_tiler = _FakeDynamicTiler()
    processor.max_num_tiles = 12
    processor.num_image_token = 5
    return processor


def test_nano_nemotron_vl_keeps_dynamic_image_tiler_with_tile_cap():
    processor = _new_dummy_image_processor()

    text, image_inputs = processor._preprocess_image(
        text=["question <image> answer"],
        images=[object()],
        max_num_tiles=12,
    )

    assert processor.dynamic_tiler.called
    assert text == ["question <repl:7:1> answer"]
    assert image_inputs["num_tokens_per_image"] == [7]
    assert image_inputs["imgs_sizes"] == [(16, 16)]


def test_nano_nemotron_vl_max_num_tiles_one_uses_static_image_path(monkeypatch):
    processor = _new_dummy_image_processor()

    def static_images_to_pixel_values_lst(images, max_num_tiles):
        assert max_num_tiles == 1
        assert len(images) == 1
        return [torch.zeros(1, 3, 16, 16)]

    monkeypatch.setattr(
        processor,
        "_images_to_pixel_values_lst",
        static_images_to_pixel_values_lst,
    )

    text, image_inputs = processor._preprocess_image(
        text=["frame <image>"],
        images=[object()],
        max_num_tiles=1,
    )

    assert not processor.dynamic_tiler.called
    assert text == ["frame <repl:5:1>"]
    assert image_inputs["image_num_patches"].tolist() == [1]


@pytest.mark.parametrize(
    ("orig_w", "orig_h", "expected_size"),
    [
        (480, 320, (608, 416)),
        (640, 424, (608, 416)),
        (1152, 720, (640, 384)),
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


def test_nano_nemotron_vl_native_video_replaces_context_tokens_only():
    repl = NanoNemotronVLProcessor.get_video_repl(
        tokens_per_frame=[2, 1],
        frames_indices=[0, 1, 2, 3],
        frame_duration_ms=500,
        tokenizer=_SeparatorTokenizer(),
        img_start_token_ids=[101],
        img_end_token_ids=[102],
        img_context_token_ids=[103],
        video_temporal_patch_size=2,
    )

    assert repl.full == [200, 101, 103, 103, 102, 201, 101, 103, 102]
    assert repl.is_embed is not None
    assert repl.is_embed(None, repl.full).tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_nano_nemotron_vl_evs_placeholder_tokens_spread_after_first_frame():
    tokens_per_frame = compute_placeholder_tokens_per_frame(
        tokens_per_frame=8,
        num_frames=4,
        q=0.5,
    )
    assert tokens_per_frame == [8, 3, 3, 2]
    assert sum(tokens_per_frame) == 16

    assert compute_placeholder_tokens_per_frame(
        tokens_per_frame=8,
        num_frames=4,
        q=0.9,
    ) == [8, 0, 0, 0]
    assert compute_placeholder_tokens_per_frame(
        tokens_per_frame=64,
        num_frames=8,
        q=0.5,
    ) == [64, 28, 28, 28, 27, 27, 27, 27]


def test_nano_nemotron_vl_detects_audio_context_in_text_and_tokens():
    processor = object.__new__(NanoNemotronVLMultiModalProcessor)
    object.__setattr__(processor, "info", _PromptInfo())

    assert processor._prompt_has_audio_context(f"before {AUDIO_CONTEXT} after")
    assert processor._prompt_has_audio_context([100, 901, 902, 101])
    assert not processor._prompt_has_audio_context("video only")
    assert not processor._prompt_has_audio_context([100, 901, 101, 902])


def test_nano_nemotron_vl_final_video_embeddings_merge_text_and_video(
    monkeypatch,
):
    monkeypatch.setattr(
        nano_module,
        "cached_tokenizer_from_config",
        lambda model_config: _SeparatorTokenizer(),
    )

    model = object.__new__(NemotronH_Nano_VL_V2)
    object.__setattr__(model, "model_config", object())
    object.__setattr__(model, "_img_start_token_ids", [101])
    object.__setattr__(model, "_img_end_token_ids", [102])
    object.__setattr__(model, "_img_context_token_ids", [103])
    object.__setattr__(
        model,
        "get_language_model",
        lambda: _EmbeddingLanguageModel(),
    )

    video_embeddings = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
    )
    result = model._create_final_video_embeddings(
        video_embeddings=video_embeddings,
        tokens_per_frame=[2, 1],
        frames_indices=[0, 1, 2, 3],
        frame_duration_ms=500,
        video_temporal_patch_size=2,
    )

    repl_token_ids = torch.tensor([200, 101, 103, 103, 102, 201, 101, 103, 102])
    expected = torch.stack((repl_token_ids.float(), -repl_token_ids.float()), dim=-1)
    expected[[2, 3, 7]] = video_embeddings
    assert torch.equal(result, expected)


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
