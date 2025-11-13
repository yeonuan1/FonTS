from torch import Tensor, nn
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    T5EncoderModel, T5Tokenizer
)
import os
from huggingface_hub import snapshot_download


class HFEmbedder(nn.Module):
    """
    - CLIP 쪽: tokenizer는 검증된 openai/clip-vit-large-patch14 사용.
               text_encoder 가중치는 black-forest-labs/FLUX.1-dev 의
               'text_encoder/' 서브폴더만 스냅샷 받아서 로드.
    - T5 쪽  : tokenizer_2, text_encoder_2 서브폴더를 스냅샷 받아서 로드.

    util.py 에서 HFEmbedder(...)를 부를 때 subfolder="text_encoder_2"를
    넘기면 자동으로 T5 경로로 분기합니다. (subfolder 인자는 여기서 pop되어
    transformers.*.from_pretrained 에 중복 전달되지 않습니다)
    """
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        super().__init__()

        # util.load_t5(...) 가 넘겨주는 subfolder 힌트로 분기
        subfolder_hint = hf_kwargs.pop("subfolder", None)
        use_t5 = (subfolder_hint == "text_encoder_2") or \
                 ("text_encoder_2" in (version or ""))

        self.max_length = max_length
        self.is_clip = not use_t5
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            # 1) 토크나이저: 안정적인 CLIP L/14 사용
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                model_max_length=max_length,
                use_fast=False,
            )

            # 2) 텍스트 인코더: FLUX.1-dev 의 text_encoder 서브폴더만 받기
            local_dir = snapshot_download(
                repo_id="black-forest-labs/FLUX.1-dev",
                allow_patterns=["text_encoder/*"],
            )
            text_enc_dir = os.path.join(local_dir, "text_encoder")

            # subfolder 인자는 transformers로 전달하지 않음
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                text_enc_dir,
                **hf_kwargs
            )

        else:
            # 1) FLUX.1-dev 의 tokenizer_2, text_encoder_2 서브폴더만 받기
            local_dir = snapshot_download(
                repo_id="black-forest-labs/FLUX.1-dev",
                allow_patterns=["tokenizer_2/*", "text_encoder_2/*"],
            )
            tok2_dir = os.path.join(local_dir, "tokenizer_2")
            enc2_dir = os.path.join(local_dir, "text_encoder_2")

            # 2) T5 토크나이저 / 인코더 로드
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                tok2_dir,
                model_max_length=max_length,
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                enc2_dir,
                **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
