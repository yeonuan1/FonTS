"""
infer_fonts.py
학습된 FonTS (mT5 + SigLIP + LoRA) 모델 
SCA를 선택적으로 적용
cd /home/shaush/FonTS-main/flux+SCA-both/src

**12,13,14줄은 없어도 됌**
python -m flux.infer_fonts \
    --checkpoint_dir "/home/shaush/FonTS-main/flux+SCA-both/src/outputs/step-6000" \
    --prompt_html "A photo of <b>bears</b> walking down the street" \
    --prompt_plain "A photo of bears walking down the street" \
    --style_image "/home/shaush/FonTS/ATR-bench/multi_letters/lego_average.png" \
    --sca_checkpoint_path "/home/shaush/FonTS/checkpoints/checkpoint-both.safetensors" \
    --ip_scale 0.8 \
    --width 512 \
    --height 512 \
    --seed 456
"""

import argparse
import torch
import yaml
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from einops import rearrange, repeat
import numpy as np
import re

from safetensors.torch import load_file as load_sft
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# --- 기존 파일에서 로직 import ---
# 유틸리티
from flux.util import load_flow_model, load_ae
from flux.model import Flux
from flux.modules.layers import (
    # IPDoubleStreamBlockProcessor, (결합 클래스로 대체됨)
    # IPSingleStreamBlockProcessor, (결합 클래스로 대체됨)
    ImageProjModel,
    DoubleStreamBlockLoraProcessor, # (신규) LoRA 전용 임포트
    SingleStreamBlockLoraProcessor, # (신규) LoRA 전용 임포트
    IPDoubleStreamBlockLoraProcessor, # (신규) 결합 클래스 임포트
    IPSingleStreamBlockLoraProcessor, # (신규) 결합 클래스 임포트
)
# 훈련 스크립트에서 로더 가져오기
from flux.train import (
    load_mt5_with_etc_tokens,
    load_siglip,
    TextProjection,
    # (신규) LoRA 부착 함수 임포트
    apply_lora_to_model, 
)
# 샘플링
from flux.sampling import get_noise, get_schedule, denoise, unpack


@torch.inference_mode()
def load_ip_adapter_components(flux_model: Flux, sca_checkpoint_path: str, device: str):
    """
    SCA 가중치를 로드하고 모델을 패치하는 대신,
    추론에 필요한 컴포넌트들(인코더, 프로젝션)과
    SCA 가중치를 반환합니다.
    """
    print(f"Loading SCA/IP-Adapter from: {sca_checkpoint_path}")
    checkpoint = load_sft(sca_checkpoint_path, device=device)
    
    # --- 모델을 직접 패치하는 대신, SCA 가중치만 추출 ---
    
    # 1. IP-Adapter 어댑터 가중치
    ip_adapter_state_dict = {
        k: v for k, v in checkpoint.items() if k.startswith("ip_adapter.")
    }
    if not ip_adapter_state_dict:
        print("⚠️ WARNING: SCA 체크포인트에서 'ip_adapter.' 가중치를 찾을 수 없습니다.")

    # 2. 이미지 인코더 및 프로젝션 로드
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device, dtype=torch.float16).eval()
    clip_image_processor = CLIPImageProcessor()

    improj = ImageProjModel(4096, 768, 4).to(device, dtype=torch.bfloat16).eval()
    image_proj_state_dict = {
        k.replace("image_proj.", ""): v for k, v in checkpoint.items() if k.startswith("image_proj.")
    }
    improj.load_state_dict(image_proj_state_dict, strict=True)
    print("✅ SCA/IP-Adapter Image Encoder and Projection loaded.")
    
    # 3. 인코더, 프로세서, 이미지 프로젝션, IP 가중치 반환
    return image_encoder, clip_image_processor, improj, ip_adapter_state_dict

@torch.inference_mode()
def get_image_proj(
    image_prompt: Image.Image, 
    image_encoder: CLIPVisionModelWithProjection, 
    clip_image_processor: CLIPImageProcessor, 
    improj: ImageProjModel,
    device: str
):
    """
    XFluxPipeline.get_image_proj 로직
    """
    image_prompt_tensors = clip_image_processor(
        images=image_prompt,
        return_tensors="pt"
    ).pixel_values
    image_prompt_tensors = image_prompt_tensors.to(image_encoder.device, dtype=torch.float16)
    
    image_prompt_embeds = image_encoder(
        image_prompt_tensors
    ).image_embeds.to(
        device=device, dtype=torch.bfloat16,
    )
    image_proj = improj(image_prompt_embeds)
    return image_proj

@torch.inference_mode()
def main(args):
    device = args.device
    
    # --- 1. Config 로드 ---
    print(f"Loading config from checkpoint directory: {args.checkpoint_dir}")
    config_path = Path(args.checkpoint_dir) / "config.yaml"
    if not config_path.exists():
        print(f"오류: {config_path}에서 config.yaml을 찾을 수 없습니다.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    cfg_model = config['model']

    # --- 2. 기본 모델 로드 ---
    flux_model = load_flow_model(cfg_model['flux_model_type'], device).eval()
    ae = load_ae(cfg_model['flux_model_type'], device).eval()

    # --- 3. 훈련된 텍스트 컴포넌트 로드 ---
    t5_model, t5_tokenizer = load_mt5_with_etc_tokens(
        cfg_model['mt5_model_name'], cfg_model['etc_tokens'], device="cpu"
    )
    siglip_model, siglip_tokenizer = load_siglip(
        cfg_model['siglip_model_name'], device="cpu"
    )
    text_proj = TextProjection(input_dim=768, output_dim=4096).to(device, dtype=torch.bfloat16).eval()

    # --- 4. (수정됨) SCA 및 LoRA 로직 ---
    
    image_proj = None
    neg_image_proj = None
    sca_weights = {}
    
    if args.style_image:
        # 4a. SCA/IP-Adapter 사용 시
        if not args.sca_checkpoint_path:
            print("오류: --style_image를 사용하려면 --sca_checkpoint_path가 필요합니다.")
            return
        
        # SCA 컴포넌트와 *가중치* 로드
        sca_encoder, sca_processor, sca_improj, sca_weights = load_ip_adapter_components(
            flux_model, args.sca_checkpoint_path, device
        )
        
        # 스타일 이미지 임베딩
        style_pil = Image.open(args.style_image).convert("RGB")
        image_proj = get_image_proj(
            style_pil, sca_encoder, sca_processor, sca_improj, device
        )
        neg_image_proj = torch.zeros_like(image_proj) # 네거티브 텐서 생성

        # (신규) *결합된* (LoRA + SCA) 어텐션 프로세서 부착
        print("Attaching combined (LoRA + SCA) processors...")
        lora_attn_procs = {}
        total_replaced = 0
        rank = cfg_model['lora_rank']
        hidden_size = flux_model.hidden_size
        
        for name, processor in flux_model.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
                if name.startswith("double_blocks") and layer_index % 2 == 0: 
                    lora_attn_procs[name] = IPDoubleStreamBlockLoraProcessor(
                        dim=hidden_size, rank=rank, dtype=torch.bfloat16
                    ).to(device)
                    total_replaced += 1
                elif name.startswith("single_blocks") and layer_index % 4 == 0:
                    lora_attn_procs[name] = IPSingleStreamBlockLoraProcessor(
                        dim=hidden_size, rank=rank, dtype=torch.bfloat16, mlp_ratio=flux_model.params.mlp_ratio
                    ).to(device)
                    total_replaced += 1
                else:
                    lora_attn_procs[name] = processor
            else:
                lora_attn_procs[name] = processor
        
        flux_model.set_attn_processor(lora_attn_procs)
        print(f"✅ Attached {total_replaced} combined (LoRA + SCA) processors.")

    else:
        # 4b. SCA 미사용 시 (LoRA만 사용)
        print("Attaching standard (LoRA-only) processors...")
        # train.py의 apply_lora_to_model 함수를 사용하여 LoRA 프로세서 부착
        apply_lora_to_model(flux_model, cfg_model, device)

    # --- 5. 훈련된 가중치 적용 (프로세서 부착 *이후*) ---
    print("Loading trained weights from checkpoint...")
    
    # 5a. LoRA 가중치 로드
    lora_path = Path(args.checkpoint_dir) / "flux_lora.safetensors"
    if lora_path.exists():
        # strict=False를 사용하여 SCA 프로세서가 LoRA 가중치만 선택적으로 로드하도록 함
        missing, unexpected = flux_model.load_state_dict(load_sft(lora_path, device=device), strict=False)
        print(f"✅ Loaded LoRA weights from: {lora_path}")
        if 'ip_adapter.' in "".join(unexpected):
             print("ℹ️ (Expected) Ignoring IP-Adapter weights while loading LoRA.")
    else:
        print("⚠️ LoRA weights not found. LoRA is NOT loaded.")

    # 5b. (신규) SCA 가중치 로드 (SCA 사용 시)
    if args.style_image and sca_weights:
        missing, unexpected = flux_model.load_state_dict(sca_weights, strict=False)
        print(f"✅ Loaded SCA/IP-Adapter weights.")
        if 'lora' in "".join(unexpected):
            print("ℹ️ (Expected) Ignoring LoRA weights while loading SCA.")

    # 5c. Text Projection 가중치
    proj_path = Path(args.checkpoint_dir) / "text_projection.safetensors"
    text_proj.load_state_dict(load_sft(proj_path, device=device))
    print(f"✅ Loaded Text Projection weights from: {proj_path}")

    # 5d. ETC 토큰 임베딩 가중치
    embed_path = Path(args.checkpoint_dir) / "t5_etc_embeddings.safetensors"
    t5_model.get_input_embeddings().load_state_dict(load_sft(embed_path, device="cpu"))
    print(f"✅ Loaded ETC Token weights into mT5 model.")
    

    # 모델을 GPU로 이동
    t5_model.to(device)
    siglip_model.to(device)
    text_proj.to(device)

    # --- 6. 프롬프트 인코딩 (train.py 로직과 유사) ---
    print("Encoding prompts...")
    
    # (A) SigLIP (vec)
    siglip_inputs = siglip_tokenizer(
        [args.prompt_plain], padding="max_length", max_length=64, 
        truncation=True, return_tensors="pt"
    ).to(device)
    vec_cond = siglip_model(**siglip_inputs).pooler_output.to(dtype=torch.bfloat16)

    # (B) mT5 (txt)
    t5_inputs = t5_tokenizer(
        [args.prompt_html], padding="max_length", max_length=512, 
        truncation=True, return_tensors="pt"
    ).to(device)
    t5_embeds = t5_model.get_input_embeddings()(t5_inputs.input_ids)
    txt_cond_768 = t5_model(
        inputs_embeds=t5_embeds,
        attention_mask=t5_inputs.attention_mask
    ).last_hidden_state.to(dtype=torch.bfloat16)

    # (C) Text Projection
    txt_cond = text_proj(txt_cond_768)

    # --- 네거티브 프롬프트 ---
    neg_siglip_inputs = siglip_tokenizer(
        [args.neg_prompt_plain], padding="max_length", max_length=64, 
        truncation=True, return_tensors="pt"
    ).to(device)
    neg_vec_cond = siglip_model(**neg_siglip_inputs).pooler_output.to(dtype=torch.bfloat16)
    
    neg_t5_inputs = t5_tokenizer(
        [args.neg_prompt_html], padding="max_length", max_length=512, 
        truncation=True, return_tensors="pt"
    ).to(device)
    neg_t5_embeds = t5_model.get_input_embeddings()(neg_t5_inputs.input_ids)
    neg_txt_cond_768 = t5_model(
        inputs_embeds=neg_t5_embeds,
        attention_mask=neg_t5_inputs.attention_mask
    ).last_hidden_state.to(dtype=torch.bfloat16)
    neg_txt_cond = text_proj(neg_txt_cond_768)

    # 인코더 오프로드
    t5_model.cpu()
    siglip_model.cpu()
    torch.cuda.empty_cache()

    # --- 7. 노이즈 및 ID 텐서 준비 ---
    width, height = args.width, args.height
    batch_size = 1

    x = get_noise(batch_size, height, width, device, torch.bfloat16, args.seed)
    
    img_cond = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    h, w = x.shape[2] // 2, x.shape[3] // 2
    
    img_ids_template = torch.zeros(h, w, 3, device=device)
    img_ids_template[..., 1] = img_ids_template[..., 1] + torch.arange(h, device=device)[:, None]
    img_ids_template[..., 2] = img_ids_template[..., 2] + torch.arange(w, device=device)[None, :]
    img_ids = repeat(img_ids_template, "h w c -> b (h w) c", b=batch_size).to(torch.bfloat16)

    txt_ids = torch.zeros(
        txt_cond.shape[0], txt_cond.shape[1], 3, device=device, dtype=torch.bfloat16
    )
    neg_txt_ids = torch.zeros(
        neg_txt_cond.shape[0], neg_txt_cond.shape[1], 3, device=device, dtype=torch.bfloat16
    )
    
    timesteps = get_schedule(
        args.num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=True,
    )
    
    # --- 8. 샘플링 (Denoise) ---
    print(f"Generating image for prompt: '{args.prompt_html}'")

    for i in tqdm(range(args.num_images_per_prompt)):
        seed = args.seed + i
        torch.manual_seed(seed)
        x = get_noise(batch_size, height, width, device, torch.bfloat16, seed)
        img_cond = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        
        output_patched = denoise(
            model=flux_model,
            img=img_cond,
            img_ids=img_ids,
            txt=txt_cond,
            txt_ids=txt_ids,
            vec=vec_cond,
            neg_txt=neg_txt_cond,
            neg_txt_ids=neg_txt_ids,
            neg_vec=neg_vec_cond,
            timesteps=timesteps,
            guidance=args.guidance,
            true_gs=args.true_gs,
            timestep_to_start_cfg=args.timestep_to_start_cfg,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=args.ip_scale,
            neg_ip_scale=args.neg_ip_scale
        )

        # --- 9. Unpack 및 저장 ---
        output_latent = unpack(output_patched.float(), height, width)
        output_image = ae.decode(output_latent)
        
        output_image = output_image.clamp(-1, 1)
        output_image = rearrange(output_image[0], "c h w -> h w c")
        output_pil = Image.fromarray((127.5 * (output_image + 1.0)).cpu().byte().numpy())
        
        os.makedirs(args.output_dir, exist_ok=True)
        safe_prompt = re.sub(r'[<>:"/\\|?*]', '_', args.prompt_html)[:50]
        output_path = os.path.join(args.output_dir, f"{safe_prompt}_{seed}.png")
        output_pil.save(output_path)
        print(f"✅ Image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="훈련된 체크포인트가 저장된 디렉토리 (e.g., ./outputs/step-2000)"
    )
    parser.add_argument(
        "--prompt_html", type=str, required=True,
        help="mT5용 프롬프트 (ETC 토큰 포함). 예: 'A photo of <i>hello</i> world'"
    )
    parser.add_argument(
        "--prompt_plain", type=str, required=True,
        help="SigLIP용 프롬프트 (ETC 토큰 없음). 예: 'A photo of hello world'"
    )
    
    parser.add_argument(
        "--style_image", type=str, default=None,
        help="SCA/IP-Adapter를 위한 스타일 레퍼런스 이미지 경로"
    )
    parser.add_argument(
        "--sca_checkpoint_path", type=str, default=None,
        help="SCA/IP-Adapter 체크포인트 경로 (e.g., checkpoint-both.safetensors)"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0, help="SCA(IP-Adapter) 강도"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0, help="네거티브 SCA(IP-Adapter) 강도"
    )

    parser.add_argument(
        "--neg_prompt_html", type=str, default="",
        help="네거티브 mT5 프롬프트 (ETC 토큰)"
    )
    parser.add_argument(
        "--neg_prompt_plain", type=str, default="",
        help="네거티브 SigLIP 프롬프트 (일반 텍스트)"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="inference_results", help="이미지 저장 경로"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="이미지 너비"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="이미지 높이"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="샘플링 스텝 수"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="시드"
    )
    parser.add_argument(
        "--guidance", type=float, default=4.0, help="가이던스 스케일 (flux-dev용)"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="True-CFG 가이던스"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=0, help="True-CFG 시작 스텝"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="프롬프트당 생성할 이미지 수"
    )
    parser.add_argument(
        "--device", type=str, default="cuda"
    )
    
    args = parser.parse_args()
    main(args)