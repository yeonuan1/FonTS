# train.py
"""
**initiate training**
cd /home/shaush/FonTS-main/flux+SCA-both/src
accelerate launch -m flux.train --config flux/config.yaml

**resume from checkpoint**
cd /home/shaush/FonTS-main/flux+SCA-both/src
accelerate launch -m flux.train \
    --config flux/config.yaml \
    --resume_from_checkpoint /home/shaush/FonTS-main/flux+SCA-both/src/outputs/step-6000
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import (
    T5Tokenizer, T5EncoderModel, AutoProcessor, SiglipTextModel,
    get_cosine_schedule_with_warmup
)
import math
import os
import numpy as np
import json
import yaml
import argparse
from PIL import Image
from einops import rearrange, repeat
from tqdm.auto import tqdm
from pathlib import Path

from accelerate import Accelerator
import wandb
from peft import LoraConfig
from flux.util import load_flow_model, load_ae
from flux.modules.layers import (
    DoubleStreamBlockLoraProcessor, 
    SingleStreamBlockLoraProcessor, 
    timestep_embedding
)

# Helper Functions/Classes
class TextProjection(nn.Module): 
    def __init__(self, input_dim=768, output_dim=4096, dtype=None): 
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim, dtype=dtype),
            nn.LayerNorm(output_dim, dtype=dtype),
        )
    def forward(self, x):
        return self.proj(x)

class TCDataset(Dataset):
    def __init__(self, metadata_path: str, image_base_path: str = './', resolution: int = 512):
        self.metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))
        self.image_base_path = image_base_path
        self.resolution = resolution

    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        item = self.metadata[idx]
        prompt_plain = item['text_plain']
        prompt_html = item['text_html']
        image_path = os.path.join(self.image_base_path, item['id'] + '.jpg')
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
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

def load_mt5_with_etc_tokens(model_name: str, etc_tokens: list, device="cpu"): 
    print(f"Loading mT5 model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5EncoderModel.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': etc_tokens})
    print(f"Added {len(etc_tokens)} new ETC tokens to mT5 tokenizer.")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized mT5 model embeddings to {len(tokenizer)}.")
    return model.to(device, dtype=torch.bfloat16), tokenizer 

def load_siglip(model_name: str, device="cpu"): 
    print(f"Loading SigLIP model: {model_name}")
    tokenizer = AutoProcessor.from_pretrained(model_name).tokenizer
    model = SiglipTextModel.from_pretrained(model_name)
    return model.to(device, dtype=torch.bfloat16), tokenizer 


def apply_lora_to_model(model, cfg_model: dict, device):
    rank = cfg_model['lora_rank']
    hidden_size = model.hidden_size
    target_modules = set(cfg_model.get('lora_target_modules', []))

    double_block_targets = {
        "img_attn.qkv", "img_attn.proj", "txt_attn.qkv", 
        "txt_attn.proj", "img_mlp.0", "img_mlp.2", 
        "txt_mlp.0", "txt_mlp.2"
    }
    single_block_targets = {"linear1", "linear2"}

    apply_to_double = any(t in target_modules for t in double_block_targets)
    apply_to_single = any(t in target_modules for t in single_block_targets)

    if not (apply_to_double or apply_to_single):
        print("⚠️ WARNING: 'use_lora: true'이지만 'lora_target_modules'가 비어있거나 알려진 타겟과 일치하지 않습니다. LoRA 레이어가 적용되지 않습니다.")
        return

    if apply_to_double:
        print(f"✅ Config 'lora_target_modules' 감지: DoubleStreamBlocks에 LoRA(rank={rank})를 적용합니다.")
    if apply_to_single:
        print(f"✅ Config 'lora_target_modules' 감지: SingleStreamBlocks에 LoRA(rank={rank})를 적용합니다.")

    lora_attn_procs = {}
    total_replaced = 0

    for name, processor in model.attn_processors.items():
        if apply_to_double and name.startswith("double_blocks"):
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                dim=hidden_size, rank=rank
            ).to(device, dtype=torch.bfloat16)
            total_replaced += 1
        elif apply_to_single and name.startswith("single_blocks"):
            lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                dim=hidden_size, rank=rank
            ).to(device, dtype=torch.bfloat16)
            total_replaced += 1
        else:
            lora_attn_procs[name] = processor
            
    if total_replaced > 0:
        model.set_attn_processor(lora_attn_procs)
    
    print(f"✅ Config 기반으로 총 {total_replaced}개의 프로세서를 LoRA 버전으로 교체했습니다.")

def set_trainable_text_parts(t5_model, text_proj, etc_tokens: list):
    text_proj.requires_grad_(True)
    print("✅ Text Projection (768→4096): Unfrozen")

    t5_embed_layer = t5_model.get_input_embeddings()
    num_new_tokens = len(etc_tokens)

    if num_new_tokens > 0:
        embedding_weight = t5_embed_layer.weight
        embedding_weight.requires_grad = True
        
        old_token_mask = torch.ones_like(embedding_weight, dtype=torch.bool)
        old_token_mask[-num_new_tokens:] = False

        old_token_mask = old_token_mask.to(embedding_weight.device)

        def zero_grad_hook(grad):
            mask_on_device = old_token_mask.to(grad.device)
            grad[mask_on_device] = 0
            return grad
        
        embedding_weight.register_hook(zero_grad_hook)
        print(f"✅ mT5: {num_new_tokens} ETC-Tokens Unfrozen (using gradient hook).")
    
    return t5_embed_layer

# 3. Trainer Class 

class Trainer:
    def __init__(self, config_path: str, resume_from_checkpoint: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cfg_model = self.config['model']
        self.cfg_data = self.config['data']
        self.cfg_train = self.config['training']
        self.cfg_log = self.config['logging']
        
        self.resume_from = resume_from_checkpoint
        self.global_step = 0
        self.epoch = 0
        self.resume_step_in_epoch = 0
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg_train['gradient_accumulation_steps'],
            mixed_precision=self.cfg_train['mixed_precision'],
            log_with="wandb" if self.cfg_log.get('use_wandb') else None,
        )

        if self.accelerator.is_main_process and self.cfg_log.get('use_wandb', False):
            self.accelerator.init_trackers(
                project_name=self.cfg_log['wandb_project'],
                config=self.config,
                resume="allow", 
                init_kwargs={"wandb": {"name": self.cfg_log['wandb_run_name']}}
            )
        
        self.device = self.accelerator.device
        
        self.setup_models()
        self.setup_data()
        self.setup_optimization()
        # self.prepare_for_training()
    
    def setup_models(self):
        self.accelerator.print("Setting up models...")
        
        self.flux_model = load_flow_model(self.cfg_model['flux_model_type'], self.device).to(torch.bfloat16)
        self.ae = load_ae(self.cfg_model['flux_model_type'], self.device).to(torch.bfloat16)
        
        self.t5_model, self.t5_tokenizer = load_mt5_with_etc_tokens(
            self.cfg_model['mt5_model_name'], self.cfg_model['etc_tokens'], device="cpu"
        )
        self.siglip_model, self.siglip_tokenizer = load_siglip(
            self.cfg_model['siglip_model_name'], device="cpu"
        )
        self.text_proj = TextProjection(
            input_dim=768, 
            output_dim=4096
        ).to("cpu", dtype=torch.bfloat16)
        
        if self.cfg_model['use_lora']:
            apply_lora_to_model(
                self.flux_model, 
                self.cfg_model, 
                self.device
            ) 

        self.flux_model.requires_grad_(False)
        self.ae.eval().requires_grad_(False)
        self.t5_model.eval().requires_grad_(False)
        self.siglip_model.eval().requires_grad_(False)
        self.text_proj.requires_grad_(False)
        self.accelerator.print("Froze all base model weights.")

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
            
        self.t5_embed_layer = set_trainable_text_parts(
            self.t5_model, self.text_proj, self.cfg_model['etc_tokens']
        )
        
        if self.cfg_train.get('gradient_checkpointing', False):
            self.flux_model.gradient_checkpointing = True
            self.accelerator.print("✅ Gradient Checkpointing enabled for Flux model.")
 
        self.ae.eval()
        self.siglip_model.eval()
        self.t5_model.eval()

        
    def setup_data(self):
        self.accelerator.print("Setting up datasets...")
        train_dataset = TCDataset(
            metadata_path=self.cfg_data['metadata_path'], 
            image_base_path=self.cfg_data['image_base_path'],
            resolution=self.cfg_data.get('resolution', 512)
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
        
        lora_params = [p for p in self.flux_model.parameters() if p.requires_grad]
        proj_params = [p for p in self.text_proj.parameters() if p.requires_grad]
        t5_embed_params = [p for p in self.t5_embed_layer.parameters() if p.requires_grad]
        
        trainable_params = lora_params + proj_params + t5_embed_params

        if not trainable_params:
            self.accelerator.print("⚠️ WARNING: No trainable parameters found!")
        
        num_etc_tokens = len(self.cfg_model['etc_tokens'])
        t5_hidden_size = self.t5_embed_layer.embedding_dim
        actual_t5_trainable_params = num_etc_tokens * t5_hidden_size
        full_t5_embed_params = sum(p.numel() for p in t5_embed_params)

        self.accelerator.print(f"📊 Trainable parameters:")
        self.accelerator.print(f"  - Flux (LoRA): {sum(p.numel() for p in lora_params):,}")
        self.accelerator.print(f"  - mT5 (ETC):   {actual_t5_trainable_params:,} (Hook applied to {full_t5_embed_params:,} total params)")
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
        self.ae.to(self.device)
        
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
        self.siglip_model.to(self.device)
        with torch.no_grad():
            siglip_inputs = self.siglip_tokenizer(
                prompts_plain, padding="max_length", max_length=64, 
                truncation=True, return_tensors="pt"
            ).to(self.device)
            vec_cond = self.siglip_model(**siglip_inputs).pooler_output.to(dtype=torch.bfloat16)
        self.siglip_model.cpu() 

        self.t5_model.to(self.device)
        t5_inputs = self.t5_tokenizer(
            prompts_html, padding="max_length", max_length=512, 
            truncation=True, return_tensors="pt"
        ).to(self.device)
        
        t5_embeds = self.t5_embed_layer(t5_inputs.input_ids)
        
        with torch.no_grad():
            txt_cond_768 = self.t5_model(
                inputs_embeds=t5_embeds,
                attention_mask=t5_inputs.attention_mask
            ).last_hidden_state.to(dtype=torch.bfloat16)
        self.t5_model.cpu() 
        # --- End Offloading ---
        
        with self.accelerator.autocast():
            txt_cond = self.text_proj(txt_cond_768)
            
            txt_ids = torch.zeros(
                txt_cond.shape[0], txt_cond.shape[1], 3, device=self.device, dtype=torch.bfloat16
            )
            img_cond = rearrange(zt, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            h, w = zt.shape[2] // 2, zt.shape[3] // 2
            img_ids_template = torch.zeros(h, w, 3, device=self.device)
            img_ids_template[..., 1] = img_ids_template[..., 1] + torch.arange(h, device=self.device)[:, None]
            img_ids_template[..., 2] = img_ids_template[..., 2] + torch.arange(w, device=self.device)[None, :]
            img_ids = repeat(img_ids_template, "h w c -> b (h w) c", b=batch_size).to(torch.bfloat16)

            noise_pred = self.flux_model(
                img=img_cond,
                img_ids=img_ids,
                txt=txt_cond,
                txt_ids=txt_ids,
                timesteps=t,
                y=vec_cond,
                guidance=guidance_tensor,
            )

            noise_pred = rearrange(
                noise_pred, 
                "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
                h=h, w=w, ph=2, pw=2
            )

            loss = F.mse_loss(noise_pred.float(), epsilon.float())

        return loss

    def train(self):
        self.accelerator.print("\n" + "="*80)
        self.accelerator.print("Starting training...")
        self.accelerator.print("="*80 + "\n")

        self.prepare_for_training()

        # --- 체크포인트 로드 ---
        if self.resume_from:
            if Path(self.resume_from).exists():
                try:
                    self.load_checkpoint(self.resume_from)
                except FileNotFoundError as e:
                    self.accelerator.print(f"🔥 Error loading checkpoint: {e}")
                    self.accelerator.print("Starting from scratch...")
                    self.epoch = 0 # 로드 실패 시 0부터 시작
            else:
                self.accelerator.print(f"⚠️ Checkpoint path not found: {self.resume_from}. Starting from scratch.")
                self.epoch = 0
        # -----------------------------

        # --- 루프 시작점을 self.epoch로 변경 ---
        for epoch in range(self.epoch, self.cfg_train['num_epochs']):
            self.epoch = epoch # 현재 에포크 저장
            
            self.flux_model.train()
            self.text_proj.train()
            self.t5_embed_layer.train()

            progress_bar = tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch+1}/{self.cfg_train['num_epochs']}",
                initial=self.resume_step_in_epoch
            )
            
            for step, batch in enumerate(progress_bar):
                if step == 0 and self.resume_step_in_epoch > 0:
                    self.resume_step_in_epoch = 0

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
    
    # --- load_checkpoint 메서드 ---
    def load_checkpoint(self, checkpoint_dir: str):
        self.accelerator.print(f"🔄 Resuming from checkpoint: {checkpoint_dir}")
        try:
            from safetensors.torch import load_file as load_sft
            
            lora_path = Path(checkpoint_dir) / "flux_lora.safetensors"
            proj_path = Path(checkpoint_dir) / "text_projection.safetensors"
            embed_path = Path(checkpoint_dir) / "t5_etc_embeddings.safetensors"

            if lora_path.exists():
                unwrapped_flux = self.accelerator.unwrap_model(self.flux_model)
                unwrapped_flux.load_state_dict(load_sft(lora_path, device=self.device), strict=False)
                self.accelerator.print("✅ Loaded LoRA weights.")
            
            if proj_path.exists():
                unwrapped_proj = self.accelerator.unwrap_model(self.text_proj)
                unwrapped_proj.load_state_dict(load_sft(proj_path, device=self.device), strict=False)
                self.accelerator.print("✅ Loaded Text Projection weights.")

            if embed_path.exists():
                unwrapped_t5_embed = self.accelerator.unwrap_model(self.t5_embed_layer)
                unwrapped_t5_embed.load_state_dict(load_sft(embed_path, device=self.device), strict=False)
                self.accelerator.print("✅ Loaded ETC Token weights.")

        except ImportError:
             self.accelerator.print("⚠️ safetensors not found, skipping inference weight loading.")
        
        # --- (신규) 2. 옵티마이저 및 스케줄러 상태 로드 ---
        opt_path = Path(checkpoint_dir) / "optimizer.pt"
        sch_path = Path(checkpoint_dir) / "scheduler.pt"
        
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
            self.accelerator.print("✅ Loaded optimizer state.")
        else:
            self.accelerator.print("⚠️ optimizer.pt not found. Resuming with fresh optimizer state.")
            
        if sch_path.exists():
            self.lr_scheduler.load_state_dict(torch.load(sch_path, map_location=self.device))
            self.accelerator.print("✅ Loaded scheduler state.")
        else:
            self.accelerator.print("⚠️ scheduler.pt not found. Resuming with fresh scheduler state.")

        # --- (유지) 3. 훈련 진행 상태 (스텝, 에포크) 로드 ---
        progress_path = Path(checkpoint_dir) / "training_state.pt"
        if progress_path.exists():
            progress_state = torch.load(progress_path, map_location="cpu")
            self.global_step = progress_state.get('global_step', 0)
            self.epoch = progress_state.get('epoch', 0)
            
            # --- (유지) 데이터로더 건너뛰기 로직 ---
            steps_per_epoch = len(self.train_dataloader)
            self.resume_step_in_epoch = self.global_step % steps_per_epoch
            
            if self.resume_step_in_epoch > 0:
                self.accelerator.print(f"🔄 Resuming epoch {self.epoch} from step {self.resume_step_in_epoch}/{steps_per_epoch}")
                self.accelerator.skip_first_batches(self.train_dataloader, self.resume_step_in_epoch)
            
            self.accelerator.print(f"✅ Resumed training state: global_step={self.global_step}, epoch={self.epoch}")
        else:
            self.accelerator.print("⚠️ training_state.pt not found. Resuming step/epoch from 0.")
            self.resume_step_in_epoch = 0

    # ---save_checkpoint ---
    def save_checkpoint(self, final=False):
        if not self.accelerator.is_main_process:
            return

        step_name = f"step-{self.global_step}"
        if final: step_name = "final"
        save_dir = Path(self.cfg_log['output_dir']) / step_name
        save_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.print(f"\n💾 Saving checkpoint to {save_dir}...")

        torch.save(self.optimizer.state_dict(), save_dir / "optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), save_dir / "scheduler.pt")

        # --- (유지) 2. 훈련 진행 상태 저장 ---
        progress_state = {
            'global_step': self.global_step,
            'epoch': self.epoch
        }
        torch.save(progress_state, save_dir / "training_state.pt")
        
        # --- (유지) 3. 추론용 가중치 저장 ---
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
            
        self.accelerator.print(f"✅ Checkpoint (resumption + inference) saved.")


# 4. Script Execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from (e.g., ./outputs/step-2000)"
    )
    # ------------------------------
    
    args = parser.parse_args()
    config_path = args.config

    trainer = Trainer(
        config_path=str(config_path),
        resume_from_checkpoint=args.resume_from_checkpoint 
    )
    trainer.train()

if __name__ == "__main__":
    main()