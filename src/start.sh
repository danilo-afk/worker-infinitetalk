#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode
comfy-manager-set-mode offline || echo "worker-ltx-video - Could not set ComfyUI-Manager network_mode" >&2

# ============ Auto-download modelos para Network Volume ============
VOLUME="/runpod-volume"
if [ -d "$VOLUME" ]; then
  echo "worker-ltx-video: Network volume detectado em $VOLUME"

  # Evita corrida entre múltiplos workers escrevendo os mesmos modelos no volume.
  LOCK_FILE="$VOLUME/.model-bootstrap.lock"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -w 1800 9; then
      echo "worker-ltx-video: timeout aguardando lock de bootstrap ($LOCK_FILE)" >&2
      exit 1
    fi
    echo "worker-ltx-video: lock de bootstrap adquirido"
  else
    echo "worker-ltx-video: flock não encontrado; bootstrap seguirá sem lock" >&2
  fi

  # Versão do modelo: 2.3 (LTX-2.3 22B, default) ou 2.0 (LTX-2 19B, legado).
  # A mesma imagem serve os dois endpoints; o endpoint escolhe via env.
  LTX_MODEL_VERSION="${LTX_MODEL_VERSION:-2.3}"
  echo "worker-ltx-video: LTX_MODEL_VERSION=$LTX_MODEL_VERSION"

  # Default serverless: FP8 para bootstrap rápido e evitar worker unhealthy por cold-start longo.
  # Para usar o checkpoint full, defina LTX_CKPT_NAME (ex.: ltx-2.3-22b-distilled-1.1.safetensors).
  if [ "$LTX_MODEL_VERSION" = "2.0" ]; then
    CKPT_NAME="${LTX_CKPT_NAME:-ltx-2-19b-distilled-fp8.safetensors}"
  else
    CKPT_NAME="${LTX_CKPT_NAME:-ltx-2.3-22b-distilled-fp8.safetensors}"
  fi

  case "$CKPT_NAME" in
    # ===== LTX-2 (19B) — legado =====
    ltx-2-19b-distilled.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors"
      CKPT_MIN=40000000000
      ;;
    ltx-2-19b-distilled.full-43285058186.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled.safetensors"
      CKPT_MIN=40000000000
      ;;
    ltx-2-19b-dev-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors"
      CKPT_MIN=25000000000
      ;;
    ltx-2-19b-distilled-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8.safetensors"
      CKPT_MIN=8500000000
      ;;
    # ===== LTX-2.3 (22B) — atual =====
    ltx-2.3-22b-distilled-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-distilled-fp8.safetensors"
      CKPT_MIN=27000000000
      ;;
    ltx-2.3-22b-dev-fp8.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors"
      CKPT_MIN=27000000000
      ;;
    ltx-2.3-22b-distilled-1.1.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-1.1.safetensors"
      CKPT_MIN=43000000000
      ;;
    ltx-2.3-22b-dev.safetensors)
      CKPT_URL="https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors"
      CKPT_MIN=43000000000
      ;;
    *)
      echo "worker-ltx-video: LTX_CKPT_NAME inválido: $CKPT_NAME" >&2
      exit 1
      ;;
  esac
  CKPT="$VOLUME/models/checkpoints/$CKPT_NAME"
  GEMMA_DIR="$VOLUME/models/text_encoders/gemma-3-fp8"
  GEMMA_OFFICIAL_DIR="$VOLUME/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"
  GEMMA_MODEL="$GEMMA_DIR/model.safetensors"
  TOKENIZER="$GEMMA_DIR/tokenizer.model"

  # Tamanhos mínimos esperados (bytes) - detecta downloads incompletos.
  # O valor do checkpoint depende do arquivo escolhido (dev vs distilled).
  GEMMA_MIN=8500000000

  validate_safetensors_coverage() {
    local file="$1"
    python - "$file" <<'PY'
import json
import os
import struct
import sys

path = sys.argv[1]
size = os.path.getsize(path)
if size < 8:
    raise SystemExit("arquivo menor que 8 bytes")

with open(path, "rb") as f:
    header_len_raw = f.read(8)
    if len(header_len_raw) != 8:
        raise SystemExit("falha ao ler tamanho do header")
    header_len = struct.unpack("<Q", header_len_raw)[0]
    if header_len <= 0 or (8 + header_len) > size:
        raise SystemExit(
            f"header inválido: header_len={header_len}, file_size={size}"
        )
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise SystemExit("header incompleto")

try:
    header = json.loads(header_bytes)
except Exception as e:
    raise SystemExit(f"header JSON inválido: {e}")

max_end = 0
for key, value in header.items():
    if key == "__metadata__":
        continue
    if not isinstance(value, dict):
        raise SystemExit(f"tensor {key} inválido: entrada não é dict")
    data_offsets = value.get("data_offsets")
    if not isinstance(data_offsets, list) or len(data_offsets) != 2:
        raise SystemExit(f"tensor {key} sem data_offsets válidos")
    start, end = data_offsets
    if not isinstance(start, int) or not isinstance(end, int):
        raise SystemExit(f"tensor {key} offsets não inteiros")
    if start < 0 or end < start:
        raise SystemExit(f"tensor {key} offsets inválidos: {start}, {end}")
    max_end = max(max_end, end)

required_size = 8 + header_len + max_end
if required_size > size:
    raise SystemExit(
        f"arquivo incompleto: required_size={required_size}, file_size={size}"
    )
PY
  }

  check_size() {
    local file="$1" min="$2"
    if [ -f "$file" ]; then
      local size
      size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
      if [ "$size" -lt "$min" ]; then
        echo "worker-ltx-video: $file corrompido (${size} bytes < ${min}), re-baixando..."
        rm -f "$file"
        return 1
      fi
      if [[ "$file" == *.safetensors ]]; then
        if ! validate_safetensors_coverage "$file" >/tmp/worker_safetensors_check.log 2>&1; then
          echo "worker-ltx-video: $file inválido (safetensors), re-baixando..."
          cat /tmp/worker_safetensors_check.log >&2 || true
          rm -f "$file"
          return 1
        fi
      fi
    else
      return 1
    fi
    return 0
  }

  # Download rápido via hf_hub_download + hf_transfer (100-200 MB/s vs ~13 MB/s do wget).
  hf_fast_download() {
    local file="$1" url="$2"
    HF_HUB_ENABLE_HF_TRANSFER=1 python - "$file" "$url" <<'PYHF'
import os, sys, shutil, tempfile
file, url = sys.argv[1], sys.argv[2]
try:
    rest = url.split("huggingface.co/", 1)[1]
    repo_part, path_part = rest.split("/resolve/", 1)
    rev, filename = path_part.split("/", 1)
except Exception as e:
    print("worker-ltx-video: hf parse falhou:", e); sys.exit(3)
try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("worker-ltx-video: huggingface_hub ausente:", e); sys.exit(4)
tmp = tempfile.mkdtemp(dir=os.path.dirname(file) or "/tmp")
try:
    p = hf_hub_download(repo_id=repo_part, filename=filename, revision=rev, local_dir=tmp)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    shutil.move(p, file)
    print("worker-ltx-video: hf_transfer OK ->", file)
except Exception as e:
    print("worker-ltx-video: hf_hub_download falhou:", e); sys.exit(5)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PYHF
  }

  download_with_validation() {
    local file="$1" min="$2" url="$3" label="$4"
    local max_attempts=3
    local attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
      echo "worker-ltx-video: Baixando ${label} (tentativa ${attempt}/${max_attempts})..."
      mkdir -p "$(dirname "$file")"
      rm -f "$file"
      # 1) rápido: hf_hub_download + hf_transfer
      if hf_fast_download "$file" "$url" && check_size "$file" "$min"; then
        return 0
      fi
      rm -f "$file"
      # 2) fallback: wget (nunca fica pior que o método antigo)
      if wget --progress=dot:giga -O "$file" "$url" && check_size "$file" "$min"; then
        return 0
      fi
      echo "worker-ltx-video: Falha ao validar ${label} na tentativa ${attempt}" >&2
      attempt=$((attempt + 1))
    done
    echo "worker-ltx-video: ERRO ao baixar ${label} após ${max_attempts} tentativas." >&2
    return 1
  }

  resolve_download_url() {
    local label="$1"
    shift
    local candidate
    for candidate in "$@"; do
      [ -z "$candidate" ] && continue
      local ok=1
      local code="n/a"
      if command -v curl >/dev/null 2>&1; then
        code=$(curl -L -s -o /dev/null -w "%{http_code}" --range 0-0 "$candidate" || true)
        if [ "$code" = "200" ] || [ "$code" = "206" ]; then
          ok=0
        fi
      elif command -v wget >/dev/null 2>&1; then
        if wget --spider -q "$candidate"; then
          ok=0
          code="200"
        else
          code="wget_fail"
        fi
      else
        echo "worker-ltx-video: nem curl nem wget disponíveis para validar URL" >&2
        return 1
      fi

      if [ "$ok" -eq 0 ]; then
        echo "$candidate"
        return 0
      fi
      echo "worker-ltx-video: ${label} URL indisponível (${code}): $candidate" >&2
    done
    return 1
  }

  # ===== Pesos InfiniteTalk (WanVideoWrapper) — todos NÃO-gated =====
  # Estrutura de pastas espelha os paths do workflow oficial do Kijai. Paths podem
  # precisar de ajuste fino na 1ª validação pelo /run (loaders do WanVideoWrapper).
  DIFF="$VOLUME/models/diffusion_models"
  mkdir -p "$DIFF/WanVideo/InfiniteTalk" "$DIFF/WanVideo/Lightx2v" \
           "$VOLUME/models/vae/wanvideo" "$VOLUME/models/clip_vision" \
           "$VOLUME/models/text_encoders" "$VOLUME/models/loras/WanVideo/Lightx2v" \
           "$VOLUME/models/wav2vec2" "$VOLUME/models/MelBandRoFormer"

  dl() { # dl <dest> <min_bytes> <url> <label>
    if ! check_size "$1" "$2"; then
      download_with_validation "$1" "$2" "$3" "$4" || echo "worker-infinitetalk: $4 indisponível"
    fi
  }

  KJ="https://huggingface.co/Kijai/WanVideo_comfy/resolve/main"
  # Modelo base Wan2.1-I2V-14B (Q8 GGUF, city96) + InfiniteTalk single (fp16, Kijai)
  dl "$DIFF/WanVideo/wan2.1-i2v-14b-480p-Q8_0.gguf" 8000000000 \
     "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf" "Wan2.1-I2V-14B Q8"
  dl "$DIFF/WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk_Single_fp16.safetensors" 4000000000 \
     "$KJ/InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors" "InfiniteTalk Single fp16"
  # Text encoder umt5-xxl + VAE + clip_vision + distill LoRA
  dl "$VOLUME/models/text_encoders/umt5-xxl-enc-bf16.safetensors" 8000000000 \
     "$KJ/umt5-xxl-enc-bf16.safetensors" "umt5-xxl enc"
  dl "$VOLUME/models/vae/wanvideo/Wan2_1_VAE_bf16.safetensors" 100000000 \
     "$KJ/Wan2_1_VAE_bf16.safetensors" "Wan VAE"
  dl "$VOLUME/models/clip_vision/clip_vision_h.safetensors" 900000000 \
     "$KJ/clip_vision_h.safetensors" "clip_vision_h"
  dl "$VOLUME/models/loras/WanVideo/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" 100000000 \
     "$KJ/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" "Lightx2v distill LoRA"
  # Áudio: wav2vec2-chinese + MelBandRoformer (separação de voz)
  dl "$VOLUME/models/wav2vec2/wav2vec2-chinese-base_fp16.safetensors" 100000000 \
     "$KJ/wav2vec2-chinese-base_fp16.safetensors" "wav2vec2-chinese"
  dl "$VOLUME/models/MelBandRoFormer/MelBandRoformer_fp16.safetensors" 100000000 \
     "$KJ/MelBandRoformer_fp16.safetensors" "MelBandRoformer"

  # ===== LongCat-Video-Avatar-1.5 (melhor versão do InfiniteTalk; Whisper/8-step distill) =====
  # Modelo bf16 (31.7GB, cabe em 48GB VRAM) + distill LoRA. Nós: WanVideoLongCatAvatarExtendEmbeds
  # + WanVideoSamplerv2/Schedulerv2 (longcat_distill_euler). VAE/umt5/wav2vec já baixados acima.
  mkdir -p "$DIFF/LongCat" "$VOLUME/models/loras/LongCat"
  dl "$DIFF/LongCat/LongCat-Avatar-15_bf16.safetensors" 20000000000 \
     "$KJ/LongCat/LongCat-Avatar-15_bf16.safetensors" "LongCat-Avatar-15 bf16"
  dl "$VOLUME/models/loras/LongCat/LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors" 500000000 \
     "$KJ/LongCat/LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors" "LongCat distill LoRA"

  echo "worker-infinitetalk: Modelos prontos no volume!"
else
  echo "worker-ltx-video: Sem network volume, usando modelos do container"
fi
# ===================================================================

echo "worker-ltx-video: Starting ComfyUI"

: "${COMFY_LOG_LEVEL:=DEBUG}"
: "${COMFY_STARTUP_LOG:=/tmp/comfyui.log}"
: "${GPU_READY_MAX_ATTEMPTS:=180}"
: "${GPU_READY_SLEEP_SECONDS:=2}"
mkdir -p "$(dirname "$COMFY_STARTUP_LOG")"
: > "$COMFY_STARTUP_LOG"
echo "worker-ltx-video: startup log em $COMFY_STARTUP_LOG"

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_MODULE_LOADING=LAZY

print_gpu_snapshot() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "worker-ltx-video: nvidia-smi snapshot:"
    nvidia-smi -L || true
    nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
  else
    echo "worker-ltx-video: nvidia-smi não encontrado" >&2
  fi
}

print_comfy_failure_excerpt() {
  local log_file="$1"
  echo "worker-ltx-video: últimas linhas do ComfyUI log:" >&2
  if [ -f "$log_file" ]; then
    tail -n 120 "$log_file" >&2 || true
  else
    echo "worker-ltx-video: log não encontrado em $log_file" >&2
  fi
}

wait_for_gpu_ready() {
  local attempt=1
  while [ "$attempt" -le "$GPU_READY_MAX_ATTEMPTS" ]; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      if nvidia-smi -L >/dev/null 2>&1; then
        echo "worker-ltx-video: GPU pronta (tentativa ${attempt}/${GPU_READY_MAX_ATTEMPTS})"
        return 0
      fi
    else
      # fallback sem nvidia-smi
      return 0
    fi
    echo "worker-ltx-video: aguardando GPU ficar disponível (${attempt}/${GPU_READY_MAX_ATTEMPTS})..."
    sleep "$GPU_READY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
  echo "worker-ltx-video: GPU indisponível após ${GPU_READY_MAX_ATTEMPTS} tentativas" >&2
  return 1
}

start_comfy_supervisor() {
  local comfy_args=("$@")
  local max_fast_failures="${COMFY_MAX_FAST_FAILURES:-5}"
  local fast_failure_window_s="${COMFY_FAST_FAILURE_WINDOW_S:-45}"
  (
    local attempt=1
    local fast_failures=0
    while true; do
      local started_at
      started_at=$(date +%s)
      echo "worker-ltx-video: iniciando ComfyUI (attempt ${attempt})"
      python -u /comfyui/main.py "${comfy_args[@]}" >> "$COMFY_STARTUP_LOG" 2>&1
      local code=$?
      local ended_at
      ended_at=$(date +%s)
      local runtime_s=$((ended_at - started_at))
      echo "worker-ltx-video: ComfyUI saiu com código ${code} (attempt ${attempt}, runtime ${runtime_s}s)" | tee -a "$COMFY_STARTUP_LOG"
      print_comfy_failure_excerpt "$COMFY_STARTUP_LOG"

      if [ "$runtime_s" -le "$fast_failure_window_s" ]; then
        fast_failures=$((fast_failures + 1))
      else
        fast_failures=0
      fi

      if [ "$fast_failures" -ge "$max_fast_failures" ]; then
        echo "worker-ltx-video: ComfyUI falhou rapidamente ${fast_failures} vezes; abortando worker para evitar loop infinito" | tee -a "$COMFY_STARTUP_LOG" >&2
        exit 1
      fi

      attempt=$((attempt + 1))
      sleep 4
      wait_for_gpu_ready || true
    done
  ) &
}

print_gpu_snapshot
wait_for_gpu_ready

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    start_comfy_supervisor --disable-auto-launch --disable-metadata --lowvram --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout

    echo "worker-ltx-video: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    start_comfy_supervisor --disable-auto-launch --disable-metadata --lowvram --verbose "${COMFY_LOG_LEVEL}" --log-stdout

    echo "worker-ltx-video: Starting RunPod Handler"
    python -u /handler.py
fi

