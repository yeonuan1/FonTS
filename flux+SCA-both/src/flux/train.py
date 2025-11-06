<<<<<<< HEAD
# train.py
=======
# -*- coding: utf-8 -*-
>>>>>>> theirs/main

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
<<<<<<< HEAD
from transformers import (
    T5Tokenizer, T5EncoderModel, AutoProcessor, SiglipTextModel,
    get_cosine_schedule_with_warmup
)
=======
from transformers import AutoTokenizer # ETC-Token 처리를 위해 토크나이저 사용 가정
>>>>>>> theirs/main
import math
import os
import numpy as np
import json
<<<<<<< HEAD
import yaml # PyYAML (pip install pyyaml)
import argparse
from PIL import Image
from einops import rearrange, repeat
from tqdm.auto import tqdm
from pathlib import Path

from accelerate import Accelerator
import wandb

# flux imports
from flux.util import load_flow_model, load_ae
from flux.modules.layers import (
    DoubleStreamBlockLoraProcessor, 
    SingleStreamBlockLoraProcessor, 
    timestep_embedding
)

# Helper Functions/Classes 

class TextProjection(nn.Module): # <-- 수정된 코드
    def __init__(self, input_dim=768, output_dim=4096, dtype=None): # 1. dtype 받기
        super().__init__()
        self.proj = nn.Sequential(
            # vvv 2. 받은 dtype을 nn.Linear와 LayerNorm에 전달 vvv
            nn.Linear(input_dim, output_dim, dtype=dtype),
            nn.LayerNorm(output_dim, dtype=dtype),
        )
    def forward(self, x):
        return self.proj(x)

class TCDataset(Dataset):
=======
from PIL import Image
import random

from .sampling import prepare
from .util import load_flow_model, load_ae

os.environ['AE'] = '/data1/FonTS/flux+SCA-both/src/models/ae.safetensors'
os.environ['FLUX_DEV'] = '/data1/FonTS/flux+SCA-both/src/models/flux1-dev.safetensors'


"""
 실제 $\text{Flux}$ 모델 로딩 함수(load_ae, load_flow_model, load_t5, load_clip) 및 
 $\text{Flux}$ 클래스 정의가 외부 파일(utils.py, model.py 등)에 정확히 존재하고 호출 가능함을 가정
"""

# ====================================================================
# 1. 파라미터 및 더미 클래스 정의 (외부 모듈을 모방)
# ====================================================================

# 파라미터 정의 (하드웨어 및 훈련 효율성을 고려하여 BATCH_SIZE=1 설정 유지)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "flux-dev"
LEARNING_RATE = 1e-5
BATCH_SIZE = 1 # 메모리 제약을 고려하여 1로 설정
NUM_EPOCHS = 10
# ⭐ 추가: Guidance Distillation Strength 설정
GUIDANCE_STRENGTH = 7.0 # 일반적인 값 (4.0 ~ 10.0 사이에서 테스트 필요)

# --- T5/mT5 로딩 함수 정의 (실제 모듈을 모방한 Dummy 클래스) ---
MT5_MODEL_NAME = "google/mt5-base" # mT5 사용을 위해 로딩 이름 정의
ETC_TOKEN_ID_START = 50000 

class TextProjection(nn.Module):
    """mT5 임베딩(768차원)을 T5-XXL 호환(4096차원)으로 변환"""
    def __init__(self, input_dim=768, output_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),  # 안정성 향상
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 768) - mT5 임베딩
        Returns:
            (batch, seq_len, 4096) - T5-XXL 호환 임베딩
        """
        return self.proj(x)

class DummyHFEmbedder:
    """실제 mT5/CLIP 인코더 모듈을 대체"""
    def __init__(self, version, max_length, torch_dtype):
        self.output_dim = 768  # mT5/CLIP-Base의 일반적인 임베딩 차원 가정
        # 실제 T5/CLIP 모듈을 로드했다고 가정하고, 임베딩을 위한 더미 레이어를 정의
        self.hf_module = torch.nn.Linear(768, 768) 
        # mT5의 임베딩 레이어 이름은 'shared'일 수 있음. 여기서는 더미로 정의
        self.hf_module.shared = torch.nn.Parameter(torch.randn(50000 + 100, 768)) 
        
    def to(self, device): return self
    def eval(self): return self
    def requires_grad_(self, requires): return self
    # __call__은 prepare 함수에서 사용됨. (B, L, D) 텐서 반환 가정
    def __call__(self, text): 
        return torch.randn(len(text), 50, self.output_dim, device=DEVICE) 
    
def load_clip(device) -> DummyHFEmbedder: 
    return DummyHFEmbedder("CLIP", 77, torch.bfloat16).to(device)

def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> DummyHFEmbedder: 
    print(f"Loading mT5 model: {MT5_MODEL_NAME}")
    return DummyHFEmbedder(MT5_MODEL_NAME, max_length=max_length, torch_dtype=torch.bfloat16).to(device)

class DummyAE:
    """실제 VAE 인코더/디코더를 대체"""
    def encode(self, x): 
        # VAE는 (B, 3, H, W) -> (B, 64, H/16, W/16) (예시)로 변환
        return torch.randn(x.shape[0], 64, 32, 32, device=DEVICE) 
    def eval(self): return self
    def requires_grad_(self, requires): return self

class DummyFlux(torch.nn.Module):
    """실제 Flux(MM-DiT) 백본을 대체"""
    def __init__(self):
        super().__init__()
        # 논문의 txt_in 레이어를 모방
        self.txt_in = nn.Linear(768, 3072) 
    def forward(self, img, img_ids, txt, txt_ids, timesteps, y, **kwargs):
        return torch.randn(img.shape[0], 64, 32, 32, device=DEVICE) 
    # 실제 Flux 모델이 가지고 있어야 할 set_etc_token_trainable이 참조하는 속성
    # txt_in이 nn.Module이므로 parameters()를 통해 접근 가능함

# ====================================================================
# 2. Dataset 및 파라미터 설정 함수
# ====================================================================

class TCDataset(Dataset):
    """metadata.jsonl 및 이미지 파일을 로드하는 실제 데이터셋 로더"""
>>>>>>> theirs/main
    def __init__(self, metadata_path: str, image_base_path: str = './'):
        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        self.image_base_path = image_base_path
<<<<<<< HEAD
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        item = self.metadata[idx]
        prompt_plain = item['text_plain']
        prompt_html = item['text_html']
        image_path = os.path.join(self.image_base_path, item['id'] + '.jpg')
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() 
            image_tensor = (image_tensor / 127.5) - 1.0 
        except Exception as e:
            return None 
        return image_tensor, prompt_plain, prompt_html

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None 
    images = [item[0] for item in batch]
    prompts_plain = [item[1] for item in batch]
    prompts_html = [item[2] for item in batch]
    images_batch = torch.stack(images)
    return images_batch, prompts_plain, prompts_html

def load_mt5_with_etc_tokens(model_name: str, etc_tokens: list, device="cpu"): # <-- Load to CPU
    print(f"Loading mT5 model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': etc_tokens})
    print(f"Added {len(etc_tokens)} new ETC tokens to mT5 tokenizer.")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized mT5 model embeddings to {len(tokenizer)}.")
    # Load to CPU, not device
    return model.to(device, dtype=torch.bfloat16), tokenizer 

def load_siglip(model_name: str, device="cpu"): # <-- Load to CPU
    print(f"Loading SigLIP model: {model_name}")
    tokenizer = AutoProcessor.from_pretrained(model_name).tokenizer
    model = SiglipTextModel.from_pretrained(model_name)
    # Load to CPU, not device
    return model.to(device, dtype=torch.bfloat16), tokenizer 


def apply_lora_to_model(model, rank: int, hidden_size: int, device, dtype): # <-- 수정: dtype 인자 추가
    """
    (FIXED) Finds all relevant attention processors and replaces them
    with LoRA-enabled processors.
    """
    lora_attn_procs = {}
    total_replaced = 0

    # Iterate over the *existing* processors
    for name, processor in model.attn_processors.items():
        if name.startswith("double_blocks"):
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                dim=hidden_size, rank=rank, dtype=dtype # <-- 수정: dtype 전달
            ).to(device) # <-- .to(device)는 유지
            total_replaced += 1
        elif name.startswith("single_blocks"):
            lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                dim=hidden_size, rank=rank, dtype=dtype # <-- 수정: dtype 전달
            ).to(device) # <-- .to(device)는 유지
            total_replaced += 1
        else:
            # Keep the original processor if it's not a target
            lora_attn_procs[name] = processor
            
    if total_replaced > 0:
        model.set_attn_processor(lora_attn_procs)
    
    # Log the *actual* number replaced
    print(f"✅ Applied LoRA (rank={rank}) to {total_replaced} attention blocks.")

def set_trainable_text_parts(t5_model, text_proj, etc_tokens: list):
    """
    Correctly unfreezes the projection layer and uses a
    gradient hook to train *only* the new ETC tokens.
    """
    text_proj.requires_grad_(True)
    print("✅ Text Projection (768→4096): Unfrozen")

    t5_embed_layer = t5_model.get_input_embeddings()
    num_new_tokens = len(etc_tokens)

    if num_new_tokens > 0:
        # Get the embedding weight parameter
        embedding_weight = t5_embed_layer.weight
        
        # 1. Make the *entire* parameter require gradients
        embedding_weight.requires_grad = True
        
        # 2. Create a mask to identify the *old* tokens
        # (Everything *except* the last 'num_new_tokens')
        old_token_mask = torch.ones_like(embedding_weight, dtype=torch.bool)
        old_token_mask[-num_new_tokens:] = False

        # Ensure mask is on the same device
        old_token_mask = old_token_mask.to(embedding_weight.device)

        # 3. Register a hook that will be called during backward()
        def zero_grad_hook(grad):
            mask_on_device = old_token_mask.to(grad.device)
            # Zero out gradients for all *old* tokens
            grad[mask_on_device] = 0
            
            return grad
        
        # Register the hook on the weight tensor
        embedding_weight.register_hook(zero_grad_hook)
        
        print(f"✅ mT5: {num_new_tokens} ETC-Tokens Unfrozen (using gradient hook).")
    
    return t5_embed_layer

# 3. Trainer Class 

class Trainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cfg_model = self.config['model']
        self.cfg_data = self.config['data']
        self.cfg_train = self.config['training']
        self.cfg_log = self.config['logging']

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg_train['gradient_accumulation_steps'],
            mixed_precision=self.cfg_train['mixed_precision'],
            log_with="wandb" if self.cfg_log.get('use_wandb') else None,
        )

        if self.accelerator.is_main_process and self.cfg_log.get('use_wandb', False):
            self.accelerator.init_trackers(
                project_name=self.cfg_log['wandb_project'],
                config=self.config,
                init_kwargs={"wandb": {"name": self.cfg_log['wandb_run_name']}}
            )
        
        self.device = self.accelerator.device
        
        self.setup_models()
        self.setup_data()
        self.setup_optimization()
        self.prepare_for_training()
        self.global_step = 0

    def setup_models(self):
        self.accelerator.print("Setting up models...")
        
        # --- 1. 모델 로드 ---
        # (Flux, AE는 GPU로, 텍스트 인코더는 CPU 오프로딩을 위해 CPU로)
        self.flux_model = load_flow_model(self.cfg_model['flux_model_type'], self.device).to(torch.bfloat16)
        self.ae = load_ae(self.cfg_model['flux_model_type'], self.device).to(torch.bfloat16)
        
        self.t5_model, self.t5_tokenizer = load_mt5_with_etc_tokens(
          self.cfg_model['mt5_model_name'], self.cfg_model['etc_tokens'], device="cpu"
        )
        self.siglip_model, self.siglip_tokenizer = load_siglip(
            self.cfg_model['siglip_model_name'], device="cpu"
        )
        # self.text_proj = TextProjection(input_dim=768, output_dim=4096).to("cpu", dtype=torch.bfloat16)
        self.text_proj = TextProjection(
            input_dim=768, 
            output_dim=4096, 
            dtype=torch.bfloat16 # 3. 생성 시 dtype 명시
        ).to("cpu")
        
        # (A) LoRA를 먼저 적용
        if self.cfg_model['use_lora']:
            model_dtype = next(self.flux_model.parameters()).dtype
            apply_lora_to_model(
                self.flux_model, 
                rank=self.cfg_model['lora_rank'], 
                hidden_size=self.flux_model.hidden_size,
                device=self.device,
                dtype=model_dtype
            ) 

        # (B) 모든 기본 모델의 파라미터를 동결
        self.flux_model.requires_grad_(False)
        self.ae.eval().requires_grad_(False)
        self.t5_model.eval().requires_grad_(False)
        self.siglip_model.eval().requires_grad_(False)
        self.text_proj.requires_grad_(False)
        self.accelerator.print("Froze all base model weights.")

        # (C) LoRA, TextProj, ETC 토큰만 다시 활성화(Unfreeze)
        # (C-1) LoRA 파라미터 활성화
        if self.cfg_model['use_lora']:
            lora_count = 0
            for name, param in self.flux_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                    lora_count += 1
            if lora_count == 0:
                self.accelerator.print("⚠️ WARNING: No LoRA parameters found to unfreeze.")
            else:
                self.accelerator.print(f"✅ Unfroze {lora_count} LoRA parameters.")
            
        # (C-2) 텍스트 관련 파라미터 활성화
        self.t5_embed_layer = set_trainable_text_parts(
            self.t5_model, self.text_proj, self.cfg_model['etc_tokens']
        )
        
        if self.cfg_train.get('gradient_checkpointing', False):
            self.flux_model.gradient_checkpointing = True
            self.accelerator.print("✅ Gradient Checkpointing enabled for Flux model.")

        # (D) 비학습 모델을 eval 모드로 설정
        self.ae.eval()
        self.siglip_model.eval()
        self.t5_model.eval()
    def setup_data(self):
        self.accelerator.print("Setting up datasets...")
        train_dataset = TCDataset(
            metadata_path=self.cfg_data['metadata_path'], 
            image_base_path=self.cfg_data['image_base_path']
        )
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.cfg_train['batch_size'], 
            shuffle=True, 
            collate_fn=custom_collate_fn,
            num_workers=self.cfg_data['num_workers']
        )

    def setup_optimization(self):
        self.accelerator.print("Setting up optimizer and scheduler...")
        
        # Correctly filter for *all* params that require grad
        lora_params = [p for p in self.flux_model.parameters() if p.requires_grad]
        proj_params = [p for p in self.text_proj.parameters() if p.requires_grad]
        t5_embed_params = [p for p in self.t5_embed_layer.parameters() if p.requires_grad]
        
        trainable_params = lora_params + proj_params + t5_embed_params

        if not trainable_params:
            self.accelerator.print("⚠️ WARNING: No trainable parameters found!")
        
        # Log the *correct* counts
        self.accelerator.print(f"📊 Trainable parameters:")
        self.accelerator.print(f"  - Flux (LoRA): {sum(p.numel() for p in lora_params):,}")
        self.accelerator.print(f"  - mT5 (ETC):   {sum(p.numel() for p in t5_embed_params):,}")
        self.accelerator.print(f"  - Projection:  {sum(p.numel() for p in proj_params):,}")
        
        self.optimizer = optim.AdamW(trainable_params, lr=self.cfg_train['learning_rate'])
        
        num_training_steps = (
            len(self.train_dataloader) * self.cfg_train['num_epochs']
        ) // self.cfg_train['gradient_accumulation_steps']
        num_warmup_steps = int(num_training_steps * 0.1)
        
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def prepare_for_training(self):
        self.accelerator.print("Preparing models with accelerator...")
        
        # Prepare only the *trainable* models
        (
            self.flux_model,
            self.text_proj,
            self.t5_embed_layer,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.flux_model,
            self.text_proj,
            self.t5_embed_layer,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler
        )
        
        # AE stays on GPU, but is not prepared (no gradients)
        self.ae.to(self.device)
        # Text encoders stay on CPU, are not prepared
        
    def training_step(self, batch):
        x0_pixel, prompts_plain, prompts_html = batch
        
        with torch.no_grad():
            x0_latent = self.ae.encode(x0_pixel.to(self.device, dtype=torch.bfloat16))
            
        batch_size = x0_latent.shape[0]
        t = torch.rand(batch_size, device=self.device, dtype=torch.bfloat16)
        epsilon = torch.randn_like(x0_latent, device=self.device, dtype=torch.bfloat16)
        zt = x0_latent * (1 - t)[:, None, None, None] + epsilon * t[:, None, None, None]
        guidance_tensor = torch.full(
            (batch_size,), self.cfg_train['guidance_strength'], device=self.device, dtype=torch.bfloat16
        )

        # --- CPU Offloading ---
        # (C-1) SigLIP 2 (vec) - Move, encode, offload
        self.siglip_model.to(self.device)
        with torch.no_grad():
            siglip_inputs = self.siglip_tokenizer(
                prompts_plain, padding="max_length", max_length=64, # <-- Corrected length
                truncation=True, return_tensors="pt"
            ).to(self.device)
            vec_cond = self.siglip_model(**siglip_inputs).pooler_output.to(dtype=torch.bfloat16)
        self.siglip_model.cpu() # Offload back to CPU

        # (C-2) mT5 (txt) - Move, encode, offload
        self.t5_model.to(self.device)
        t5_inputs = self.t5_tokenizer(
            prompts_html, padding="max_length", max_length=512, 
            truncation=True, return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings from the *prepared* (GPU) embedding layer
        t5_embeds = self.t5_embed_layer(t5_inputs.input_ids)
        
        with torch.no_grad():
            txt_cond_768 = self.t5_model(
                inputs_embeds=t5_embeds,
                attention_mask=t5_inputs.attention_mask
            ).last_hidden_state.to(dtype=torch.bfloat16)
        self.t5_model.cpu() # Offload back to CPU
        # --- End Offloading ---
        with self.accelerator.autocast():
            # (C-3) Text Projection (Trainable)
            txt_cond = self.text_proj(txt_cond_768)
            
            # (C-4) ID 텐서 준비
            txt_ids = torch.zeros(
                txt_cond.shape[0], txt_cond.shape[1], 3, device=self.device, dtype=torch.bfloat16
            )
            img_cond = rearrange(zt, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            h, w = zt.shape[2] // 2, zt.shape[3] // 2
            img_ids_template = torch.zeros(h, w, 3, device=self.device)
            img_ids_template[..., 1] = img_ids_template[..., 1] + torch.arange(h, device=self.device)[:, None]
            img_ids_template[..., 2] = img_ids_template[..., 2] + torch.arange(w, device=self.device)[None, :]
            img_ids = repeat(img_ids_template, "h w c -> b (h w) c", b=batch_size).to(torch.bfloat16)

            # (D) Flux MM-DiT 순전파 (Trainable)
            noise_pred = self.flux_model(
                img=img_cond,
                img_ids=img_ids,
                txt=txt_cond,
                txt_ids=txt_ids,
                timesteps=t,
                y=vec_cond,
                guidance=guidance_tensor,
            )

            # (h, w were defined earlier as zt.shape[2] // 2, zt.shape[3] // 2)
            noise_pred = rearrange(
                noise_pred, 
                "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
                h=h, w=w, ph=2, pw=2
            )

            # (E) 손실 계산
            loss = F.mse_loss(noise_pred.float(), epsilon.float())

        return loss

    def train(self):
        self.accelerator.print("\n" + "="*80)
        self.accelerator.print("Starting training...")
        self.accelerator.print("="*80 + "\n")

        for epoch in range(self.cfg_train['num_epochs']):
            self.flux_model.train()
            self.text_proj.train()
            self.t5_embed_layer.train()

            progress_bar = tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}/{self.cfg_train['num_epochs']}"
            )
            
            for step, batch in enumerate(progress_bar):
                if batch is None:
                    continue
                
                with self.accelerator.accumulate(self.flux_model):
                    loss = self.training_step(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.global_step % self.cfg_log['log_every'] == 0:
                        lr = self.lr_scheduler.get_last_lr()[0]
                        log_data = {
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "epoch": epoch,
                        }
                        self.accelerator.log(log_data, step=self.global_step)
                        progress_bar.set_postfix(loss=loss.item())
                    
                    if self.global_step % self.cfg_log['save_every'] == 0:
                        self.save_checkpoint()

        self.accelerator.print("Training complete!")
        self.save_checkpoint(final=True)
        self.accelerator.end_training()

    def save_checkpoint(self, final=False):
        if not self.accelerator.is_main_process:
            return

        step_name = f"step-{self.global_step}"
        if final: step_name = "final"
        save_dir = Path(self.cfg_log['output_dir']) / step_name
        save_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.print(f"\n💾 Saving checkpoint to {save_dir}...")

        unwrapped_flux = self.accelerator.unwrap_model(self.flux_model)
        unwrapped_proj = self.accelerator.unwrap_model(self.text_proj)
        unwrapped_t5_embed = self.accelerator.unwrap_model(self.t5_embed_layer)
        
        flux_lora_state_dict = {
            k: v for k, v in unwrapped_flux.state_dict().items() if "lora" in k
        }
        
        try:
            from safetensors.torch import save_file
            save_file(flux_lora_state_dict, save_dir / "flux_lora.safetensors")
            save_file(unwrapped_proj.state_dict(), save_dir / "text_projection.safetensors")
            save_file(unwrapped_t5_embed.state_dict(), save_dir / "t5_etc_embeddings.safetensors")
        except ImportError:
            torch.save(flux_lora_state_dict, save_dir / "flux_lora.pt")
            torch.save(unwrapped_proj.state_dict(), save_dir / "text_projection.pt")
            torch.save(unwrapped_t5_embed.state_dict(), save_dir / "t5_etc_embeddings.pt")

        with open(save_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
            
        self.accelerator.print(f"✅ Checkpoint saved.")

# 4. Script Execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file"
    )
    args = parser.parse_args()
    
    # Ensure config path is absolute or relative to execution dir
    # This assumes config.yaml is in the same dir as train.py
    config_path = args.config

    trainer = Trainer(config_path=str(config_path))
    trainer.train()

if __name__ == "__main__":
    main()
=======

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        prompt = item['prompt']
        image_path = os.path.join(self.image_base_path, item['image'])
        
        try:
            image = Image.open(image_path).convert("RGB")
            # 이미지를 -1에서 1 사이의 Torch 텐서로 변환 (VAE 입력 형식)
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() 
            image_tensor = (image_tensor / 127.5) - 1.0 
        except Exception as e:
            # 오류 발생 시 None 반환 (collate_fn이 처리함)
            # print(f"Error loading image {image_path}: {e}")
            return None 

        return image_tensor, prompt


def to_bfloat16_dict(d):
    """딕셔너리의 모든 텐서를 bfloat16으로 변환"""
    return {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v 
            for k, v in d.items()}


def custom_collate_fn(batch):
    """Dataloader에서 이미지 로드 실패(None) 샘플을 필터링"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None 
    
    images = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    images_batch = torch.stack(images)
    
    return images_batch, prompts

class TrainingPipeline:
    """훈련 파이프라인의 모든 구성 요소를 묶는 컨테이너"""
    def __init__(self, t5, clip, ae, model, text_proj=None):
        self.t5 = t5
        self.clip = clip
        self.ae = ae
        self.model = model
        self.text_proj = text_proj

def set_etc_token_trainable(pipeline, is_trainable: bool):
    """mT5 ETC-Tokens 및 Joint Text Attention 관련 가중치만 학습 가능하도록 설정"""
    model = pipeline.model
    t5_module = pipeline.t5.hf_module

    # 1. mT5 Encoder 전체 가중치 Freeze
    for name, param in t5_module.named_parameters():
        param.requires_grad_(False)
    
    # 2. ETC-Tokens만 Unfreeze (T5 임베딩 레이어의 'shared' 파라미터 가정)
    # if hasattr(t5_module, 'shared') and isinstance(t5_module.shared, torch.Tensor):
    #     embedding_weight = t5_module.shared
    #     if embedding_weight.shape[0] > ETC_TOKEN_ID_START:
    #         embedding_weight[ETC_TOKEN_ID_START:].requires_grad_(is_trainable)
    #         print(f"mT5 Embedding: {embedding_weight.shape[0] - ETC_TOKEN_ID_START} ETC-Tokens Unfrozen.")

    # 2. ETC-Tokens만 Unfreeze 
    # mT5의 실제 임베딩 레이어 찾기
    if hasattr(t5_module, 'shared'):
        embedding_weight = t5_module.shared
        if isinstance(embedding_weight, torch.nn.Parameter):
            # Parameter인 경우
            if embedding_weight.shape[0] > ETC_TOKEN_ID_START:
                embedding_weight.requires_grad_(is_trainable)
                print(f"✅ mT5: {embedding_weight.shape[0] - ETC_TOKEN_ID_START} ETC-Tokens Unfrozen")
        elif hasattr(embedding_weight, 'weight'):
            # Embedding 레이어인 경우
            if embedding_weight.weight.shape[0] > ETC_TOKEN_ID_START:
                # 일부만 학습 가능하도록 설정하려면 hook 사용 필요
                embedding_weight.weight.requires_grad_(is_trainable)
                print(f"✅ mT5 Embedding: All tokens unfrozen (including {embedding_weight.weight.shape[0] - ETC_TOKEN_ID_START} ETC)")

    # 3. Text Projection Layer Unfreeze
    if pipeline.text_proj is not None:
        for param in pipeline.text_proj.parameters():
            param.requires_grad_(is_trainable)
        print(f"✅ Text Projection (768→4096): Unfrozen")

    # 4. Flux 모델의 Joint Text Attention (Txt-Attn) 관련 가중치 Unfreeze
    if hasattr(model, 'txt_in'):
        for param in model.txt_in.parameters():
            param.requires_grad_(is_trainable)
            
    print(f"Txt-in layer requires_grad_({is_trainable})")

# ====================================================================
# 3. 메인 훈련 함수
# ====================================================================

# def train_fonts_tc_ft_mt5(model_type=MODEL_TYPE, metadata_path='metadata.jsonl', image_base_path='./'):
#     print(f"Starting FonTS TC Fine-tuning on {DEVICE}...")

#     # --- 1. 모델 및 환경 로드 ---
#     # 실제 환경에서는 load_t5, load_clip, load_ae, load_flow_model 호출 필요
#     flux_model = load_flow_model(model_type, DEVICE) # 원본 flux 그대로 로드(4096차원)
#     ae = load_ae(model_type, DEVICE)
    
#     pipeline = TrainingPipeline(load_t5(DEVICE), load_clip(DEVICE), ae, flux_model)
    
#     # Freeze VAE 및 CLIP (Frozen)
#     pipeline.ae.eval().requires_grad_(False)
#     pipeline.clip.hf_module.eval().requires_grad_(False)
    
#     # ETC-Token 및 Txt-Attn 학습 가능하도록 설정
#     set_etc_token_trainable(pipeline, is_trainable=True) 

#     # --- 2. 데이터 준비 ---
#     train_dataset = TCDataset(metadata_path=metadata_path, image_base_path=image_base_path)
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=BATCH_SIZE, 
#         shuffle=True, 
#         collate_fn=custom_collate_fn,
#         num_workers=4
#     )
    
#     # --- 3. 옵티마이저 설정 ---
#     trainable_flux_params = [p for p in pipeline.model.parameters() if p.requires_grad]
#     trainable_t5_params = [p for p in pipeline.t5.hf_module.parameters() if p.requires_grad]
#     trainable_params = trainable_flux_params + trainable_t5_params
    
#     if not trainable_params:
#         raise ValueError("No trainable parameters found! Check set_etc_token_trainable logic.")
    
#     optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)
#     pipeline.model.train()
    
#     # --- 4. 훈련 루프 ---
#     for epoch in range(NUM_EPOCHS):
#         for step, batch_data in enumerate(train_loader):
#             if batch_data is None: 
#                 continue
                
#             x0_pixel, text_prompt = batch_data
            
#             optimizer.zero_grad()
#             x0_pixel = x0_pixel.to(DEVICE) 
            
#             # (A) VAE 인코딩 (x0_latent)
#             with torch.no_grad():
#                 x0_latent = pipeline.ae.encode(x0_pixel) 
                
#             # (B) 노이즈 샘플링 및 zt 계산
#             batch_size = x0_latent.shape[0]
#             t = torch.rand(batch_size, device=DEVICE)
#             epsilon = torch.randn_like(x0_latent, device=DEVICE)

#             # Rectified Flow Matching zt: zt = x0 * (1-t) + epsilon * t
#             zt = x0_latent * (1 - t)[:, None, None, None] + epsilon * t[:, None, None, None]

#             # (C) 조건부 텐서 준비 (mT5 및 CLIP)
#             # prepare 함수가 zt를 패치 형태로 변환하고 T5/CLIP 호출하여 조건 생성
#             inp_cond = prepare(t5=pipeline.t5, clip=pipeline.clip, img=zt, prompt=text_prompt)
            
#             # (D) Flux MM-DiT 순전파: 노이즈 예측
#             noise_pred = pipeline.model(
#                 img=inp_cond['img'],
#                 img_ids=inp_cond['img_ids'],
#                 txt=inp_cond['txt'], 
#                 txt_ids=inp_cond['txt_ids'],
#                 timesteps=t,
#                 y=inp_cond['vec'],
#             )
            
#             # (E) 손실 계산 (LCFM = MSE Loss)
#             loss = F.mse_loss(noise_pred, epsilon)
            
#             # (F) 역전파 및 가중치 업데이트
#             loss.backward()
#             optimizer.step()

#             if (step + 1) % 50 == 0:
#                 print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {step+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

#     print("mT5-based TC Fine-tuning Complete. 🇰🇷")

#     # --- 5. 모델 저장 로직 ---
#     CHECKPOINT_DIR = "checkpoints"
#     os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#     CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fonts_tc_ft_mt5_final.safetensors")

#     state = {
#         'flux_model_state_dict': pipeline.model.state_dict(),
#         't5_embedder_state_dict': pipeline.t5.hf_module.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#     }

#     # safetensors 라이브러리를 사용하여 저장하는 것이 좋습니다.
#     # [주의]: safetensors.torch.save_file 함수는 외부 라이브러리 함수이므로, torch.save로 대체
#     torch.save(state, CHECKPOINT_PATH)
#     print(f"\n✅ Checkpoint saved to: {CHECKPOINT_PATH}")


def train_fonts_tc_ft_mt5(model_type=MODEL_TYPE, metadata_path='metadata.jsonl', image_base_path='./'):
    print(f"Starting FonTS TC Fine-tuning on {DEVICE}...")

    # --- 1. 모델 로드 (원본 Flux 그대로 로드) ---
    flux_model = load_flow_model(model_type, DEVICE)  # ⚠️ 4096 차원 그대로
    ae = load_ae(model_type, DEVICE)
    t5 = load_t5(DEVICE)
    clip = load_clip(DEVICE)

    # ⭐ VAE를 bfloat16으로 변환
    ae = ae.to(torch.bfloat16)
    flux_model = flux_model.to(torch.bfloat16)
    
    # --- 2. Text Projection Layer 초기화 ⭐ ---
    text_proj = TextProjection(input_dim=768, output_dim=4096).to(DEVICE, dtype=torch.bfloat16)
    print("✅ Text Projection Layer initialized (768 → 4096)")
    
    # --- 3. Pipeline 구성 ---
    pipeline = TrainingPipeline(t5, clip, ae, flux_model, text_proj)
    
    # ⭐ Flux 전체를 먼저 Freeze
    pipeline.model.eval()
    for param in pipeline.model.parameters():
        param.requires_grad_(False)
    print("✅ Flux model fully frozen")

    # Freeze VAE 및 CLIP
    pipeline.ae.eval().requires_grad_(False)
    pipeline.clip.hf_module.eval().requires_grad_(False)
    
    # ETC-Token, Text Projection, Txt-Attn 학습 가능하도록 설정
    set_etc_token_trainable(pipeline, is_trainable=True)

    # --- 4. 데이터 준비 ---
    train_dataset = TCDataset(metadata_path=metadata_path, image_base_path=image_base_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=4
    )
    
    # --- 5. 옵티마이저 설정 (Text Projection 포함) ⭐ ---
    trainable_flux_params = [p for p in pipeline.model.parameters() if p.requires_grad]
    trainable_t5_params = [p for p in pipeline.t5.hf_module.parameters() if p.requires_grad]
    trainable_proj_params = [p for p in pipeline.text_proj.parameters() if p.requires_grad]
    
    trainable_params = trainable_flux_params + trainable_t5_params + trainable_proj_params
    
    if not trainable_params:
        raise ValueError("No trainable parameters found!")
    
    print(f"📊 Trainable parameters:")
    print(f"  - Flux (txt_in): {sum(p.numel() for p in trainable_flux_params):,}")
    print(f"  - mT5 (ETC): {sum(p.numel() for p in trainable_t5_params):,}")
    print(f"  - Projection: {sum(p.numel() for p in trainable_proj_params):,}")
    
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)
    pipeline.model.train()
    pipeline.text_proj.train()  # ⭐ Projection도 train 모드
    
    # --- 6. 훈련 루프 ---
    for epoch in range(NUM_EPOCHS):
        for step, batch_data in enumerate(train_loader):
            if batch_data is None: 
                continue
                
            x0_pixel, text_prompt = batch_data
            optimizer.zero_grad()

            # ⭐ bfloat16으로 변환
            x0_pixel = x0_pixel.to(DEVICE, dtype=torch.bfloat16)
            
            # (A) VAE 인코딩
            with torch.no_grad():
                x0_latent = pipeline.ae.encode(x0_pixel)
                
            # (B) 노이즈 샘플링 및 zt 계산
            batch_size = x0_latent.shape[0]
            t = torch.rand(batch_size, device=DEVICE, dtype=torch.bfloat16)
            epsilon = torch.randn_like(x0_latent, device=DEVICE, dtype=torch.bfloat16)
            zt = x0_latent * (1 - t)[:, None, None, None] + epsilon * t[:, None, None, None]

            # ⭐⭐ 새로운 guidance 텐서 생성 (Batch Size와 동일하게)
            guidance_tensor = torch.full(
                (batch_size,), 
                GUIDANCE_STRENGTH, # 상단에서 정의한 값
                device=DEVICE, 
                dtype=torch.bfloat16 # bfloat16으로 생성
            )

            # (C) 조건부 텐서 준비 ⭐ 수정 필요
            inp_cond = prepare(
                t5=pipeline.t5, 
                clip=pipeline.clip, 
                img=zt, 
                prompt=text_prompt,
                dtype=torch.bfloat16 
            )
            
            inp_cond = to_bfloat16_dict(inp_cond)

            # 디버깅 출력 추가
            print(f"DEBUG: inp_cond['txt'] shape before projection: {inp_cond['txt'].shape}")

            # ⭐ mT5 임베딩을 Projection Layer로 변환
            # inp_cond['txt']는 (B, L, 768) 형태
            # Text Projection (bfloat16 유지)
            txt_projected = pipeline.text_proj(inp_cond['txt'].to(torch.bfloat16))  # (B, L, 4096)
            
            # 디버깅 출력 추가
            print(f"DEBUG: txt_projected shape after projection: {txt_projected.shape}") 
            print(f"DEBUG: txt_ids shape: {inp_cond['txt_ids'].shape}") 
            print(f"DEBUG: img shape: {inp_cond['img'].shape}") 
            print(f"DEBUG: img_ids shape: {inp_cond['img_ids'].shape}")


            # (D) Flux MM-DiT 순전파
            noise_pred = pipeline.model(
                img=inp_cond['img'],
                img_ids=inp_cond['img_ids'],
                txt=txt_projected,  # ⭐ 변환된 텐서 사용
                txt_ids=inp_cond['txt_ids'],
                timesteps=t,
                y=inp_cond['vec'],
                guidance=guidance_tensor,
            )
            
            # (E) 손실 계산
            loss = F.mse_loss(noise_pred, epsilon)
            
            # (F) 역전파 및 업데이트
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {step+1}, Loss: {loss.item():.4f}")

    print("✅ mT5-based TC Fine-tuning Complete!")

    # --- 7. 모델 저장 (Text Projection 포함) ⭐ ---
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "fonts_tc_ft_mt5_final.safetensors")

    state = {
        'flux_model_state_dict': pipeline.model.state_dict(),
        't5_embedder_state_dict': pipeline.t5.hf_module.state_dict(),
        'text_projection_state_dict': pipeline.text_proj.state_dict(),  # ⭐ 추가
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(state, CHECKPOINT_PATH)
    print(f"✅ Checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    # 사용자님의 환경에 맞게 metadata 파일 경로와 이미지 기본 경로를 설정합니다.
    # 예시:
    METADATA_PATH = "/data1/FonTS/flux+SCA-both/src/flux/tc-dataset/metadata.jsonl"
    IMAGE_BASE_PATH = "/data1/FonTS/flux+SCA-both/src/flux" 
    
    try:
        train_fonts_tc_ft_mt5(
            model_type="flux-dev", 
            metadata_path=METADATA_PATH, 
            image_base_path=IMAGE_BASE_PATH
        )
    except ValueError as e:
        print(f"Error during training setup: {e}")
        print("Please ensure the Flux model parameters (context_in_dim, etc.) are correctly configured in external files (like utils.py) to match the mT5 embedding dimension (768).")
>>>>>>> theirs/main
