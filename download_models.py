from huggingface_hub import hf_hub_download
import os

# 다운로드할 로컬 디렉토리 설정
LOCAL_DIR = "/data3/FonTS/models"
os.makedirs(LOCAL_DIR, exist_ok=True)

# 1. FLUX 백본 모델 파일 (예시: flux1-dev.safetensors)
# 이 파일은 해당 저장소에 있을 가능성이 높습니다.
FLOW_MODEL_REPO = "black-forest-labs/FLUX.1-dev"
FLOW_MODEL_FILE = "flux1-dev.safetensors"

hf_hub_download(
    repo_id=FLOW_MODEL_REPO,
    filename=FLOW_MODEL_FILE,
    local_dir=LOCAL_DIR
)
print(f"✅ FLOW_MODEL 다운로드 완료: {os.path.join(LOCAL_DIR, FLOW_MODEL_FILE)}")

# 2. VAE (AutoEncoder) 파일 (예시: ae.safetensors)
AE_MODEL_REPO = "black-forest-labs/FLUX.1-dev"
AE_MODEL_FILE = "ae.safetensors"

hf_hub_download(
    repo_id=AE_MODEL_REPO,
    filename=AE_MODEL_FILE,
    local_dir=LOCAL_DIR
)
print(f"✅ VAE 다운로드 완료: {os.path.join(LOCAL_DIR, AE_MODEL_FILE)}")

# 3. mT5 텍스트 인코더 (Hugging Face transformers 라이브러리가 자동으로 처리)
# mT5는 'google/mt5-base'와 같이 저장소 이름만 알면 transformers가 자동으로 다운로드 및 캐시합니다.
# mT5는 가중치 파일 (.safetensors) 하나가 아닌 여러 구성 파일로 이루어진 폴더 구조입니다.
# 따라서, 코드를 수정하여 mT5는 로컬 경로 대신 Hugging Face 저장소 이름을 직접 지정하는 것이 더 쉽습니다.