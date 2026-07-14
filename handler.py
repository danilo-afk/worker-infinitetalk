import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import math
import os
import requests
import base64
from io import BytesIO
import websocket
import uuid
import tempfile
import socket
import traceback
import logging
import subprocess
import re
import hashlib
try:
    from PIL import Image
except ImportError:
    Image = None

from network_volume import (
    is_network_volume_debug_enabled,
    run_network_volume_diagnostics,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMFY_API_AVAILABLE_INTERVAL_MS = int(os.environ.get("COMFY_API_AVAILABLE_INTERVAL_MS", 100))
COMFY_API_AVAILABLE_MAX_RETRIES = int(os.environ.get("COMFY_API_AVAILABLE_MAX_RETRIES", 1800))
WEBSOCKET_RECONNECT_ATTEMPTS = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))
MAX_INLINE_VIDEO_BYTES = int(os.environ.get("MAX_INLINE_VIDEO_BYTES", 4_000_000))
COMFY_STARTUP_LOG = os.environ.get("COMFY_STARTUP_LOG", "/tmp/comfyui.log")
WORKFLOW_EVENT_IDLE_TIMEOUT_S = int(os.environ.get("WORKFLOW_EVENT_IDLE_TIMEOUT_S", 180))
GEMMA_NODE_IDLE_TIMEOUT_S = int(os.environ.get("GEMMA_NODE_IDLE_TIMEOUT_S", 900))
CHECKPOINT_NODE_IDLE_TIMEOUT_S = int(os.environ.get("CHECKPOINT_NODE_IDLE_TIMEOUT_S", 1200))
SAMPLER_NODE_IDLE_TIMEOUT_S = int(os.environ.get("SAMPLER_NODE_IDLE_TIMEOUT_S", 1800))
DECODE_NODE_IDLE_TIMEOUT_S = int(os.environ.get("DECODE_NODE_IDLE_TIMEOUT_S", 900))

if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    websocket.enableTrace(True)

COMFY_HOST = "127.0.0.1:8188"
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

OOM_PATTERNS = [
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"cuda.*oom", re.IGNORECASE),
    re.compile(r"cublas_status_alloc_failed", re.IGNORECASE),
    re.compile(r"memoryerror", re.IGNORECASE),
    re.compile(r"std::bad_alloc", re.IGNORECASE),
    re.compile(r"insufficient memory", re.IGNORECASE),
    re.compile(r"not enough memory", re.IGNORECASE),
    re.compile(r"killed process", re.IGNORECASE),
]

GPU_PATTERNS = [
    re.compile(r"cuda", re.IGNORECASE),
    re.compile(r"cudnn", re.IGNORECASE),
    re.compile(r"cublas", re.IGNORECASE),
    re.compile(r"nvidia", re.IGNORECASE),
    re.compile(r"device-side assert", re.IGNORECASE),
    re.compile(r"illegal memory access", re.IGNORECASE),
    re.compile(r"driver", re.IGNORECASE),
]


def _detect_image_format(blob):
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if blob[:2] == b"\xff\xd8":
        return "jpeg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    if blob[:3] == b"GIF":
        return "gif"
    return "unknown"


def _summarize_bytes(blob):
    return {
        "size_bytes": len(blob),
        "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
        "magic_hex_16": blob[:16].hex(),
        "detected_format": _detect_image_format(blob),
    }


def _get_idle_timeout_for_node_class(node_class):
    # LTX-2 usa LTXVGemmaCLIPModelLoader; LTX-2.3 usa LTXAVTextEncoderLoader.
    if node_class in ("LTXVGemmaCLIPModelLoader", "LTXAVTextEncoderLoader"):
        return GEMMA_NODE_IDLE_TIMEOUT_S
    if node_class == "CheckpointLoaderSimple":
        return CHECKPOINT_NODE_IDLE_TIMEOUT_S
    if node_class == "SamplerCustomAdvanced":
        return SAMPLER_NODE_IDLE_TIMEOUT_S
    # InfiniteTalk (WanVideoWrapper): sampler + loaders/encoders pesados (Q8 dequant,
    # block-swap CPU<->GPU, wav2vec download) ficam longos sem eventos de websocket.
    if node_class in (
        "WanVideoSampler",
        "WanVideoModelLoader",
        "MultiTalkModelLoader",
        "DownloadAndLoadWav2VecModel",
        "Wav2VecModelLoader",
        "WanVideoTextEncodeCached",
        "WanVideoVAELoader",
        "WanVideoImageToVideoMultiTalk",
        "MultiTalkWav2VecEmbeds",
    ):
        return SAMPLER_NODE_IDLE_TIMEOUT_S
    # LTX-2 usa LTXVSpatioTemporalTiledVAEDecode; LTX-2.3 usa LTXVTiledVAEDecode/LTXVAudioVAEDecode.
    if node_class in (
        "LTXVSpatioTemporalTiledVAEDecode",
        "LTXVTiledVAEDecode",
        "LTXVAudioVAEDecode",
    ):
        return DECODE_NODE_IDLE_TIMEOUT_S
    return WORKFLOW_EVENT_IDLE_TIMEOUT_S


def _probe_image_path(path):
    info = {
        "path": path,
        "exists": os.path.exists(path),
    }
    if not info["exists"]:
        return info

    with open(path, "rb") as fh:
        raw = fh.read()
    info.update(_summarize_bytes(raw))

    try:
        result = subprocess.run(
            ["file", "-b", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["file_cmd"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except Exception as exc:
        info["file_cmd_error"] = str(exc)

    if Image is None:
        info["pillow"] = "unavailable"
        return info

    try:
        with Image.open(path) as img:
            img.load()
            info["pillow"] = {
                "format": img.format,
                "mode": img.mode,
                "size": list(img.size),
            }
    except Exception as exc:
        info["pillow_error"] = f"{type(exc).__name__}: {exc}"

    return info


def _preflight_loadimage_inputs(workflow):
    load_nodes = _extract_loadimage_nodes(workflow)
    checks = []
    for node in load_nodes:
        expected = node.get("expected_image")
        if not isinstance(expected, str):
            continue
        path = os.path.join("/comfyui/input", expected)
        probe = _probe_image_path(path)
        checks.append({
            "node_id": node.get("node_id"),
            "expected_image": expected,
            "probe": probe,
        })
    return checks


def _safe_dict_keys(value):
    if isinstance(value, dict):
        return sorted(list(value.keys()))
    return []


def _build_runtime_diagnostics(parts):
    """
    Classifica mensagens de erro em categorias úteis para retorno de API.
    """
    if parts is None:
        return None

    if isinstance(parts, str):
        text_parts = [parts]
    elif isinstance(parts, (list, tuple)):
        text_parts = [str(p) for p in parts if p is not None and str(p).strip()]
    else:
        text_parts = [str(parts)]

    joined = " | ".join(text_parts).strip()
    if not joined:
        return None

    for pattern in OOM_PATTERNS:
        if pattern.search(joined):
            return {
                "category": "GPU_OOM",
                "matched": pattern.pattern,
                "message": joined[:1200],
            }

    for pattern in GPU_PATTERNS:
        if pattern.search(joined):
            return {
                "category": "GPU_RUNTIME",
                "matched": pattern.pattern,
                "message": joined[:1200],
            }

    return {
        "category": "UNKNOWN",
        "message": joined[:1200],
    }


def _read_comfy_log_tail(max_lines=120):
    if not COMFY_STARTUP_LOG or not os.path.exists(COMFY_STARTUP_LOG):
        return []
    try:
        with open(COMFY_STARTUP_LOG, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return [line.rstrip("\n") for line in lines[-max_lines:] if line.strip()]
    except Exception as exc:
        return [f"(falha ao ler log de startup: {exc})"]


def _grep_comfy_log(needle, context=6, max_hits=40):
    """Grep no arquivo INTEIRO de startup (não só tail): retorna linhas que casam
    o needle + `context` linhas ao redor (p/ pegar traceback de import de custom node)."""
    if not COMFY_STARTUP_LOG or not os.path.exists(COMFY_STARTUP_LOG):
        return []
    try:
        with open(COMFY_STARTUP_LOG, "r", encoding="utf-8", errors="replace") as fh:
            lines = [ln.rstrip("\n") for ln in fh.readlines()]
    except Exception as exc:
        return [f"(falha ao ler log: {exc})"]
    low = needle.lower()
    picked = set()
    for i, ln in enumerate(lines):
        if low in ln.lower():
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                picked.add(j)
            if len(picked) > max_hits * (context * 2 + 1):
                break
    return [lines[i] for i in sorted(picked)]


def _extract_loadimage_nodes(workflow):
    """Retorna lista de nós LoadImage com nome esperado do arquivo."""
    load_nodes = []
    if not isinstance(workflow, dict):
        return load_nodes

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type != "LoadImage":
            continue
        inputs = node.get("inputs", {})
        expected_image = inputs.get("image") if isinstance(inputs, dict) else None
        load_nodes.append({"node_id": str(node_id), "expected_image": expected_image})

    return load_nodes


def _build_workflow_node_lookup(workflow):
    lookup = {}
    if not isinstance(workflow, dict):
        return lookup
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        lookup[str(node_id)] = node.get("class_type") or "unknown"
    return lookup


def _log_ws_event(prefix, payload):
    print(f"worker-ltx-video - {prefix}:", json.dumps(payload, ensure_ascii=False))


def _resolve_model_probe(path_hint, categories):
    if not isinstance(path_hint, str) or not path_hint:
        return {"path_hint": path_hint, "resolved": None, "exists": False}

    candidates = []
    if os.path.isabs(path_hint):
        candidates.append(path_hint)
    else:
        for category in categories:
            candidates.append(os.path.join("/runpod-volume/models", category, path_hint))
            candidates.append(os.path.join("/comfyui/models", category, path_hint))

    for candidate in candidates:
        if os.path.exists(candidate):
            info = {
                "path_hint": path_hint,
                "resolved": candidate,
                "exists": True,
                "size_bytes": os.path.getsize(candidate),
            }
            try:
                with open(candidate, "rb") as fh:
                    info["sha256_16"] = hashlib.sha256(fh.read(1024 * 1024)).hexdigest()[:16]
            except Exception as exc:
                info["sha256_error"] = str(exc)
            return info

    return {"path_hint": path_hint, "resolved": None, "exists": False, "candidates": candidates}


def _preflight_gemma_loader(workflow):
    results = []
    if not isinstance(workflow, dict):
        return results

    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "LTXVGemmaCLIPModelLoader":
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        gemma_path = inputs.get("gemma_path")
        ltxv_path = inputs.get("ltxv_path")
        results.append({
            "node_id": str(node_id),
            "gemma_path": _resolve_model_probe(gemma_path, ["text_encoders", "LLM"]),
            "ltxv_path": _resolve_model_probe(ltxv_path, ["checkpoints"]),
            "max_length": inputs.get("max_length"),
        })
    return results


def _log_job_diagnostics(job_id, job_input, workflow, input_images):
    """Logs curtos para diagnosticar se I2V recebeu imagens corretamente."""
    input_keys = _safe_dict_keys(job_input)
    image_count = len(input_images) if isinstance(input_images, list) else 0
    image_names = []
    image_payload_sizes = []

    if isinstance(input_images, list):
        for image in input_images:
            if not isinstance(image, dict):
                continue
            name = image.get("name")
            payload = image.get("image")
            if isinstance(name, str):
                image_names.append(name)
            if isinstance(payload, str):
                image_payload_sizes.append(len(payload))

    load_nodes = _extract_loadimage_nodes(workflow)
    expected_names = [
        item.get("expected_image")
        for item in load_nodes
        if isinstance(item.get("expected_image"), str)
    ]

    print(
        "worker-ltx-video - Job input summary:",
        json.dumps(
            {
                "job_id": job_id,
                "input_keys": input_keys,
                "workflow_node_count": len(workflow) if isinstance(workflow, dict) else 0,
                "load_image_nodes": load_nodes,
                "images_count": image_count,
                "image_names": image_names,
                "image_payload_sizes": image_payload_sizes,
            },
            ensure_ascii=False,
        ),
    )

    if load_nodes and image_count == 0:
        print(
            "worker-ltx-video - WARNING: Workflow possui LoadImage, mas input.images veio vazio/ausente."
        )

    if expected_names and image_names:
        missing = [name for name in expected_names if name not in image_names]
        extra = [name for name in image_names if name not in expected_names]
        if missing or extra:
            print(
                "worker-ltx-video - WARNING: Nomes de imagem não batem com LoadImage:",
                json.dumps({"expected": expected_names, "received": image_names, "missing": missing, "extra": extra}),
            )


def _comfy_server_status():
    """Verifica se o servidor ComfyUI HTTP está acessível."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(ws_url, max_attempts, delay_s, initial_error):
    """Tenta reconectar ao WebSocket após desconexão."""
    print(f"worker-ltx-video - Websocket fechou: {initial_error}. Reconectando...")
    last_error = initial_error
    for attempt in range(max_attempts):
        srv_status = _comfy_server_status()
        if not srv_status["reachable"]:
            print(f"worker-ltx-video - ComfyUI HTTP inacessível – abortando reconexão")
            raise websocket.WebSocketConnectionClosedException(
                "ComfyUI HTTP unreachable during websocket reconnect"
            )
        print(f"worker-ltx-video - Tentativa {attempt + 1}/{max_attempts}...")
        try:
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)
            print(f"worker-ltx-video - Websocket reconectado.")
            return new_ws
        except (websocket.WebSocketException, ConnectionRefusedError, socket.timeout, OSError) as err:
            last_error = err
            print(f"worker-ltx-video - Tentativa {attempt + 1} falhou: {err}")
            if attempt < max_attempts - 1:
                time.sleep(delay_s)

    raise websocket.WebSocketConnectionClosedException(
        f"Falha ao reconectar. Último erro: {last_error}"
    )


# MSR: o guide/IC-LoRA deixa a FOLHA de referência (sujeitos lado a lado) "sangrar"
# nos primeiros frames de conteúdo (LTXVCropGuides só corta o latente extra, não o
# bleed). Cortamos esses frames iniciais da saída. Compensado no length do _build_msr.
_MSR_TRIM_FRAMES = int(os.environ.get("LTX_MSR_TRIM_FRAMES", "24") or 24)


def _trim_leading_frames(video_bytes, filename, n_frames):
    """Remove os primeiros `n_frames` do vídeo (frame-exato, independente de fps)."""
    if n_frames <= 0:
        return video_bytes
    src = dst = None
    try:
        src = os.path.join(tempfile.gettempdir(), f"trin_{uuid.uuid4().hex}.mp4")
        dst = os.path.join(tempfile.gettempdir(), f"trout_{uuid.uuid4().hex}.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)
        r = subprocess.run(
            ["ffmpeg", "-i", src, "-vf", f"select='gte(n\\,{n_frames})',setpts=N/FRAME_RATE/TB",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", "-y", dst],
            capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            print(f"worker-ltx-video - trim ffmpeg error: {r.stderr.decode()[:300]}")
            return video_bytes
        with open(dst, "rb") as f:
            out = f.read()
        print(f"worker-ltx-video - msr: cortados {n_frames}f iniciais (folha de referência)")
        return out
    except Exception as e:
        print(f"worker-ltx-video - trim falhou: {e}")
        return video_bytes
    finally:
        for p in (src, dst):
            if p and os.path.exists(p):
                os.remove(p)


def convert_video_to_mp4(video_bytes, filename):
    """Converte vídeo para MP4 via ffmpeg se necessário."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".mp4":
        return video_bytes, filename

    src_path = None
    mp4_path = None
    try:
        src_path = os.path.join(tempfile.gettempdir(), f"src_{uuid.uuid4().hex}{ext}")
        mp4_path = os.path.join(tempfile.gettempdir(), f"out_{uuid.uuid4().hex}.mp4")
        with open(src_path, "wb") as f:
            f.write(video_bytes)

        result = subprocess.run(
            ["ffmpeg", "-i", src_path, "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-y", mp4_path],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"worker-ltx-video - ffmpeg error: {result.stderr.decode()}")
            return video_bytes, filename

        with open(mp4_path, "rb") as f:
            mp4_bytes = f.read()

        new_filename = os.path.splitext(filename)[0] + ".mp4"
        print(f"worker-ltx-video - Convertido {filename} -> {new_filename}")
        return mp4_bytes, new_filename
    except Exception as e:
        print(f"worker-ltx-video - Conversão falhou: {e}")
        return video_bytes, filename
    finally:
        for p in [src_path, mp4_path]:
            if p and os.path.exists(p):
                os.remove(p)


_MIME_BY_EXT = {".mp4": "video/mp4", ".webm": "video/webm", ".gif": "image/gif",
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def upload_binary_artifact(job_id, payload_bytes, filename, default_ext):
    """Upload de artefato binário e retorna URL pública. Sem bucket S3 configurado
    (rp_upload devolve 'simulated_uploaded/...'), cai em base64 data URI — o
    platform_k aceita URL http OU data URI, então o worker fica autossuficiente."""
    tmp_path = None
    try:
        file_ext = os.path.splitext(filename)[1] or default_ext
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(payload_bytes)
            tmp_path = tmp.name
        url = rp_upload.upload_image(job_id, tmp_path)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
        # fallback base64 (sem S3): data URI que o platform_k baixa/decodifica
        mime = _MIME_BY_EXT.get(file_ext.lower(), "application/octet-stream")
        return f"data:{mime};base64," + base64.b64encode(payload_bytes).decode()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _has_output_node(workflow):
    """Validação mínima: workflow precisa ter ao menos um output node conhecido."""
    if not isinstance(workflow, dict):
        return False

    output_nodes = {
        "VHS_VideoCombine",
        "SaveVideo",
        "SaveWEBM",
        "SaveImage",
        "PreviewImage",
    }
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in output_nodes:
            return True
    return False


WORKFLOWS_DIR = os.environ.get("LTX_WORKFLOWS_DIR", "/workflows")
# Pontos de injeção por template (nó do prompt positivo/negativo + LoadImage no i2v).
_TEMPLATE_INJECT = {
    "t2v": {"file": "ltx23_t2v.json", "positive": "92:3", "negative": "92:4", "load_image": None,
            "length": "92:62", "fps": 24, "preprocess": None, "empty_image": "92:89"},
    "i2v": {"file": "ltx23_i2v.json", "positive": "153:132", "negative": "153:123", "load_image": "153:124",
            "length": "153:125", "fps": 24, "preprocess": "i2v:preprocess", "empty_image": None},
    # Caminho B: referência de conteúdo (IC-LoRA Ingredients) — a imagem é uma FOLHA
    # DE REFERÊNCIA (personagens/props/cenário); dims na EmptyLTXVLatentVideo (3059).
    "iclora": {"file": "ltx23_iclora.json", "positive": "2483", "negative": "2612", "load_image": "2004",
               "length": "5072", "fps": 24, "preprocess": None, "empty_image": None, "empty_latent": "3059"},
    # Caminho B (N imagens SEPARADAS): Multi-Subject Reference. Cada imagem = 1 sujeito
    # (até 4) + 1 background; PromptRelayEncode conduz o prompt; dims em INTConstant.
    "msr": {"file": "ltx23_msr.json", "prompt_relay": "99", "subjects": ["29", "40"], "background": "30",
            "width": "43", "height": "44", "length": "50", "fps": 30},
    # Talking-avatar (LongCat-Avatar-1.5, "melhor versão" — Whisper 32000): imagem (retrato)
    # + áudio (fala) -> vídeo falante. positive/negative = `positive_prompt`/`negative_prompt`
    # (WanVideoTextEncodeCached). num_frames em DOIS nós: LongCatAvatarWhisperEmbeds (7) e
    # WanVideoLongCatAvatarExtendEmbeds (12). Backup InfiniteTalk = talking_avatar_infinitetalk.json.
    "talking_avatar": {"file": "talking_avatar.json", "load_image": "1", "load_audio": "5",
                       "positive": "11", "negative": "11", "length": "7", "length2": "12",
                       "fps": 25, "resize": "2"},
}

# Teto de frames do talking-avatar. LongCat = vídeo longo NATIVO (passagem única):
# o modelo gera o vídeo inteiro num sampler só, sem segmentação no Python.
_TALKING_MAX_FRAMES = 1501  # ~60s @ 25fps

# aspect_ratio (node) -> (W, H) na mesma classe de área (~921k px), múltiplos de 32.
# t2v: seta o EmptyImage (fonte de resolução). i2v: redimensiona a imagem de entrada
# (cover, sem distorção) — a resolução do vídeo deriva dela via GetImageSize.
_ASPECT_WH = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (960, 960),
    "4:3": (1088, 816),
    "3:4": (816, 1088),
}


def _resize_cover(data_uri, w, h):
    """Redimensiona a imagem (data URI) p/ (w,h) em modo COVER (escala + crop central),
    sem distorcer o aspecto. Sem PIL, devolve a original."""
    if Image is None:
        return data_uri
    try:
        b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
        im = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        sw, sh = im.size
        scale = max(w / sw, h / sh)
        nw, nh = max(w, round(sw * scale)), max(h, round(sh * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - w) // 2, (nh - h) // 2
        im = im.crop((left, top, left + w, top + h))
        out = BytesIO()
        im.save(out, format="PNG")
        return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
    except Exception as e:
        print(f"worker-ltx-video - resize_cover falhou ({e}); usando imagem original")
        return data_uri


# Teto do lado maior da imagem i2v QUANDO não há aspect_ratio (a resolução deriva
# da imagem × 0.5 × upscale ×2 = tamanho original; imagem grande → 193 frames em alta
# res → estoura executionTimeout/VRAM). Preserva o aspecto, só reduz. Múltiplo de 32.
_LTX_MAX_LONGSIDE = int(os.environ.get("LTX_MAX_LONGSIDE", "1280") or 1280)


def _cap_longside(data_uri, cap):
    """Reduz a imagem se o lado maior passa de `cap` (preserva aspecto). Sem PIL, no-op."""
    if Image is None or cap <= 0:
        return data_uri
    try:
        b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
        im = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        sw, sh = im.size
        if max(sw, sh) <= cap:
            return data_uri
        scale = cap / max(sw, sh)
        nw, nh = (round(sw * scale) // 32) * 32 or 32, (round(sh * scale) // 32) * 32 or 32
        im = im.resize((nw, nh), Image.LANCZOS)
        out = BytesIO()
        im.save(out, format="PNG")
        print(f"worker-ltx-video - i2v sem aspect: cap {sw}x{sh} -> {nw}x{nh}")
        return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
    except Exception as e:
        print(f"worker-ltx-video - cap_longside falhou ({e}); usando imagem original")
        return data_uri


def _fit_area(data_uri, target_area=None):
    """Redimensiona a imagem p/ ~`target_area` px PRESERVANDO o aspecto, dims múltiplas
    de 32 (exigência do latente LTX). Retorna (data_uri, (w,h)). Sem PIL: devolve (orig, None)."""
    target_area = target_area or int(os.environ.get("LTX_ICLORA_AREA", str(768 * 768)))
    if Image is None:
        return data_uri, None
    try:
        b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
        im = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        sw, sh = im.size
        scale = (target_area / (sw * sh)) ** 0.5
        nw = max(32, (round(sw * scale) // 32) * 32)
        nh = max(32, (round(sh * scale) // 32) * 32)
        im = im.resize((nw, nh), Image.LANCZOS)
        out = BytesIO()
        im.save(out, format="PNG")
        return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode(), (nw, nh)
    except Exception as e:
        print(f"worker-ltx-video - fit_area falhou ({e})")
        return data_uri, None


def _neutral_plate(w, h, rgb=(210, 205, 200)):
    """Gera um plate de fundo neutro (data URI) p/ o background obrigatório do MSR.
    O MSR compõe os sujeitos sobre esse plate; o prompt descreve a cena real."""
    if Image is None:
        return None
    out = BytesIO()
    Image.new("RGB", (max(32, w), max(32, h)), rgb).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def _compose_reference_sheet(images):
    """Compõe N imagens numa FOLHA de referência (grid) — a IC-LoRA condiciona por uma
    única folha composta. Grid ~quadrado (colunas = ceil(sqrt(N))). Sem PIL: usa a 1ª."""
    if Image is None or len(images) <= 1:
        return images[0]
    try:
        import math

        ims = []
        for it in images:
            d = it["image"]
            b64 = d.split(",", 1)[1] if d.startswith("data:") else d
            ims.append(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))
        cols = math.ceil(len(ims) ** 0.5)
        rows = math.ceil(len(ims) / cols)
        cell = 512
        sheet = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
        for i, im in enumerate(ims):
            im2 = im.copy()
            im2.thumbnail((cell, cell), Image.LANCZOS)
            x = (i % cols) * cell + (cell - im2.width) // 2
            y = (i // cols) * cell + (cell - im2.height) // 2
            sheet.paste(im2, (x, y))
        out = BytesIO()
        sheet.save(out, format="PNG")
        print(f"worker-ltx-video - reference sheet: {len(ims)} imagens -> grid {cols}x{rows}")
        return {"name": "ref.png", "image": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()}
    except Exception as e:
        print(f"worker-ltx-video - compose_reference_sheet falhou ({e}); usando 1ª imagem")
        return images[0]

# LTX exige nº de frames no formato 8n+1. Teto p/ caber na VRAM (RTX 4090 24GB, 22B
# + upscaler 2-stage). Override por env LTX_MAX_FRAMES (0 = sem teto).
_LTX_MAX_FRAMES = int(os.environ.get("LTX_MAX_FRAMES", "241") or 241)


def _snap_frames(frames, cap):
    """Ajusta p/ 8n+1 (exigência do latente LTX) e aplica teto de VRAM."""
    frames = max(9, int(frames))
    if cap:
        frames = min(frames, cap)
    n = max(1, round((frames - 1) / 8))
    return n * 8 + 1


def _url_to_data_uri(url):
    """Baixa uma URL http(s) e devolve data URI base64 (para input.images)."""
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "image/png").split(";")[0]
    return f"data:{ct};base64," + base64.b64encode(resp.content).decode()


# MSR roda a 50fps (coerência de movimento). Teto de frames p/ caber em VRAM/tempo
# (~4s @ 50fps; clipes MSR são curtos por natureza, estilo "clip a clip" do tutorial).
_LTX_MSR_MAX_FRAMES = int(os.environ.get("LTX_MSR_MAX_FRAMES", "249") or 249)


def _build_msr(job_input, inj, images, prompt):
    """Monta o workflow MSR (N imagens separadas -> vídeo multi-sujeito).

    Cada imagem = 1 sujeito (até len(subjects)); gera um plate de background neutro
    (input obrigatório do LiconMSR); PromptRelayEncode conduz o prompt (global+local).
    """
    path = os.path.join(WORKFLOWS_DIR, inj["file"])
    with open(path) as f:
        wf = json.load(f)

    # aspect_ratio -> dims (INTConstant width/height); default do template se ausente.
    aspect = (job_input.get("aspect_ratio") or "").strip()
    wh = _ASPECT_WH.get(aspect)
    if wh and inj["width"] in wf and inj["height"] in wf:
        wf[inj["width"]]["inputs"]["value"], wf[inj["height"]]["inputs"]["value"] = wh
        print(f"worker-ltx-video - msr aspect {aspect} -> {wh[0]}x{wh[1]}")
    w = wf[inj["width"]]["inputs"]["value"]
    h = wf[inj["height"]]["inputs"]["value"]

    # Sujeitos -> LoadImage nós (até os slots disponíveis); nomes canônicos.
    subjects = inj["subjects"]
    upload = []
    for i, node in enumerate(subjects):
        if i < len(images) and node in wf:
            name = f"s{i + 1}.png"
            wf[node]["inputs"]["image"] = name
            upload.append({"name": name, "image": images[i]["image"]})
    n_subj = len(upload)  # nº de sujeitos de fato conectados
    # Background obrigatório: plate neutro do tamanho do vídeo (a cena vem do prompt).
    plate = _neutral_plate(w, h)
    if plate and inj["background"] in wf:
        wf[inj["background"]]["inputs"]["image"] = "bg.png"
        upload.append({"name": "bg.png", "image": plate})

    # PromptRelay: global ancora os sujeitos numa ÚNICA cena compartilhada (senão o MSR
    # tende a um split-screen, cada sujeito no seu contexto de referência); local = ação
    # do segmento (não pode ser vazio; nó exige >=1). Prefixa diretiva de unificação.
    # A CONTAGEM explícita evita multidão: "the subjects" (sem nº) faz o modelo
    # renderizar um grupo; "the two subjects" fixa exatamente N pessoas na cena.
    _num = {1: "one subject", 2: "two subjects", 3: "three subjects", 4: "four subjects"}.get(
        n_subj, f"{n_subj} subjects"
    )
    pr = inj["prompt_relay"]
    if prompt and pr in wf:
        wf[pr]["inputs"]["global_prompt"] = (
            f"exactly {_num} together in the same single frame, one shared scene, "
            f"one continuous cinematic shot, no extra people. {prompt}"
        )
        # local = ação do segmento: injeta pistas de MOVIMENTO (senão o MSR trava os
        # sujeitos = movimento "robótico"). Descreve corpo + câmera, não só a cena.
        wf[pr]["inputs"]["local_prompts"] = (
            f"{prompt}. natural fluid body movement, subtle gestures and head turns "
            "while talking, weight shifting, smooth cinematic camera motion, lifelike dynamics"
        )
    neg = job_input.get("negative_prompt")

    # Duração -> length (INTConstant), teto menor p/ MSR. +trim: gera frames extras p/
    # o corte da folha (início) não encurtar a duração pedida.
    fps = inj.get("fps") or 24
    frames = None
    if job_input.get("num_frames"):
        frames = int(job_input["num_frames"])
    elif job_input.get("duration") or job_input.get("duration_seconds"):
        frames = round(float(job_input.get("duration") or job_input.get("duration_seconds")) * fps)
    if frames and inj["length"] in wf:
        snapped = _snap_frames(frames + _MSR_TRIM_FRAMES, _LTX_MSR_MAX_FRAMES)
        wf[inj["length"]]["inputs"]["value"] = snapped
        print(f"worker-ltx-video - msr duração: {frames}f +{_MSR_TRIM_FRAMES}trim -> {snapped}f")

    print(f"worker-ltx-video - modo-prompt: msr | sujeitos={len(upload) - 1} | {w}x{h} | prompt={prompt[:60]!r}")
    return wf, upload


# --- LongCat-Avatar-1.5: builder de PASSAGEM ÚNICA (vídeo longo nativo) ---------------
_LC_WINDOW = 93   # piso de frames (janela mínima do LongCat)
_LC_NEG = ("bright tones, overexposed, static, blurred details, subtitles, worst quality, "
           "low quality, deformed, disfigured, extra fingers")
_LC_VAE = {"enable_vae_tiling": False, "tile_x": 272, "tile_y": 272,
           "tile_stride_x": 144, "tile_stride_y": 128}


def _audio_seconds(data_uri):
    """Duração (s) de um áudio base64/data-URI via ffprobe — pra derivar o comprimento
    do vídeo do próprio áudio (o vídeo acompanha a fala). None se não der."""
    if not data_uri or not isinstance(data_uri, str):
        return None
    b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    tmp = None
    try:
        raw = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(raw)
            tmp = f.name
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", tmp],
            capture_output=True, text=True, timeout=30)
        secs = float((out.stdout or "").strip())
        return secs if secs > 0 else None
    except Exception as e:
        print(f"worker-longcat - _audio_seconds falhou: {e}")
        return None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def _round_4kp1(n):
    """Wan/LongCat: num_frames válido = 4k+1 (compressão temporal 4x do VAE)."""
    k = max(1, round((n - 1) / 4))
    return k * 4 + 1


def _build_talking_avatar(job_input, inj, images, prompt):
    """Talking-avatar (LongCat-Avatar-1.5): grafo de PASSAGEM ÚNICA nativa. O LongCat é
    feito p/ vídeo longo — gera o vídeo inteiro (30-60s) num único sampler, com a
    extensão temporal INTERNA ao modelo. Sem segmentação/encadeamento no Python
    (re-encodar por janela era o gargalo). Duração deriva do áudio. Retorna (grafo, images)."""
    image_name = images[0]["name"] if images else "example.png"
    audio = job_input.get("audio")
    if isinstance(audio, dict):
        audio = [audio]
    audio_name = audio[0]["name"] if audio else "audio.wav"
    neg = job_input.get("negative_prompt") or _LC_NEG
    fps = 25  # LongCat-Avatar-1.5 é treinado a 25fps
    total = None
    if job_input.get("num_frames"):
        total = int(job_input["num_frames"])
    elif job_input.get("duration") or job_input.get("duration_seconds"):
        total = round(float(job_input.get("duration") or job_input.get("duration_seconds")) * fps)
    elif audio:
        # SEM duração pedida: deriva do PRÓPRIO ÁUDIO (vídeo acompanha a fala inteira).
        secs = _audio_seconds(audio[0].get("audio"))
        if secs:
            total = round(secs * fps)
    total = _round_4kp1(max(_LC_WINDOW, min(total or _LC_WINDOW, _TALKING_MAX_FRAMES)))
    aspect = (job_input.get("aspect_ratio") or "9:16").strip()
    # Resolução NATIVA do LongCat/Wan 480p. 9:16 retrato = melhor enquadramento p/ avatar.
    w, h = {"9:16": (480, 832), "16:9": (832, 480), "1:1": (640, 640)}.get(aspect, (480, 832))

    g = {}
    g["img"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    g["resize"] = {"class_type": "ImageResizeKJv2", "inputs": {"image": ["img", 0], "width": w,
        "height": h, "upscale_method": "lanczos", "keep_proportion": "crop", "pad_color": "0, 0, 0",
        "crop_position": "center", "divisible_by": 16, "device": "cpu"}}
    g["vae"] = {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "wanvideo/Wan2_1_VAE_bf16.safetensors",
        "precision": "bf16", "use_cpu_cache": False, "verbose": False}}
    g["enc_init"] = {"class_type": "WanVideoEncode", "inputs": {"vae": ["vae", 0], "image": ["resize", 0],
        "noise_aug_strength": 0, "latent_strength": 1, **_LC_VAE}}
    g["aud"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}
    g["whisper"] = {"class_type": "WhisperModelLoader", "inputs": {"model": "whisper_large_v3_encoder_fp16.safetensors",
        "base_precision": "fp16", "load_device": "main_device"}}
    g["embeds"] = {"class_type": "LongCatAvatarWhisperEmbeds", "inputs": {"whisper_model": ["whisper", 0],
        "audio_1": ["aud", 0], "normalize_loudness": True, "num_frames": total, "fps": fps,
        "audio_scale": 1, "audio_cfg_scale": 1, "multi_audio_type": "para"}}
    # 48GB: block_swap baixo = modelo residente na VRAM = sampler rápido (skill: 10).
    # Ajustável por /run p/ achar o ponto sem OOM de VRAM no latente longo, sem rebuild.
    blocks = int(job_input.get("blocks_to_swap") or 10)
    g["blockswap"] = {"class_type": "WanVideoBlockSwap", "inputs": {"blocks_to_swap": blocks, "offload_img_emb": False,
        "offload_txt_emb": False, "use_non_blocking": False, "vace_blocks_to_swap": 0, "prefetch_blocks": 1,
        "block_swap_debug": False}}
    g["lora"] = {"class_type": "WanVideoLoraSelect", "inputs": {"lora": "LongCat/LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors",
        "strength": 0.9, "low_mem_load": False, "merge_loras": False}}
    g["model"] = {"class_type": "WanVideoModelLoader", "inputs": {"model": "LongCat/LongCat-Avatar-15_bf16.safetensors",
        "base_precision": "bf16", "quantization": "disabled", "load_device": "offload_device",
        "attention_mode": "sdpa", "block_swap_args": ["blockswap", 0], "lora": ["lora", 0]}}
    g["text"] = {"class_type": "WanVideoTextEncodeCached", "inputs": {"model_name": "umt5-xxl-enc-bf16.safetensors",
        "precision": "bf16", "positive_prompt": prompt or "a person talking, natural expression",
        "negative_prompt": neg, "quantization": "disabled", "use_disk_cache": False, "device": "gpu"}}
    g["sched"] = {"class_type": "WanVideoSchedulerv2", "inputs": {"scheduler": "longcat_distill_euler",
        "steps": 12, "shift": 12, "start_step": 0, "end_step": -1, "enhance_hf": False}}
    # PASSAGEM ÚNICA: num_frames = total (o modelo estende internamente), 1 sampler, 1 decode.
    g["extend"] = {"class_type": "WanVideoLongCatAvatarExtendEmbeds", "inputs": {"prev_latents": ["enc_init", 0],
        "audio_embeds": ["embeds", 0], "num_frames": total, "overlap": 1, "frames_processed": 0,
        "if_not_enough_audio": "pad_with_start", "ref_frame_index": 10, "ref_mask_frame_range": 3}}
    g["sampler"] = {"class_type": "WanVideoSamplerv2", "inputs": {"model": ["model", 0], "image_embeds": ["extend", 0],
        "cfg": 1, "seed": 1, "force_offload": True, "scheduler": ["sched", 0], "text_embeds": ["text", 0],
        "add_noise_to_samples": False}}
    g["decode"] = {"class_type": "WanVideoDecode", "inputs": {"vae": ["vae", 0], "samples": ["sampler", 0],
        "normalization": "default", **_LC_VAE}}
    g["vhs"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": ["decode", 0], "audio": ["embeds", 1],
        "frame_rate": fps, "loop_count": 0, "filename_prefix": "LongCat", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True}}
    print(f"worker-longcat - talking_avatar SINGLE-PASS: {total}f (~{total/fps:.1f}s) {w}x{h}")
    return g, images


def build_workflow_from_prompt(job_input):
    """Modo-prompt: monta o workflow LTX a partir de {prompt, images/image_urls, params}.

    0 imagens -> T2V; 1+ imagens -> I2V (1ª imagem = primeiro frame). Espelha o padrão do
    kiara_new: seleciona o template pelo nº de imagens e injeta prompt/imagem.
    Retorna (workflow, images) onde images = [{name, image(base64)}] p/ upload.
    """
    prompt = (job_input.get("prompt") or "").strip()

    # Normaliza imagens: aceita `images` [{name,image}] OU `image_urls`/`reference_images` (URLs).
    images = list(job_input.get("images") or [])
    urls = job_input.get("image_urls") or job_input.get("reference_images") or []
    if isinstance(urls, str):
        urls = [urls]
    for i, u in enumerate(urls):
        if not u:
            continue
        images.append({"name": f"ref_{i}.png", "image": _url_to_data_uri(u)})

    # Talking-avatar (InfiniteTalk): áudio presente -> vídeo falante (imagem + fala).
    if job_input.get("audio"):
        return _build_talking_avatar(job_input, _TEMPLATE_INJECT["talking_avatar"], images, prompt)

    # Modo: `content_reference` (image_as_reference no node) + imagens -> IC-LoRA
    # (referência de conteúdo por FOLHA composta). Senão: 1+ img = i2v, 0 = t2v.
    if images and job_input.get("content_reference"):
        # O node escolhe pelo tipo de input: 2+ imagens SEPARADAS -> MSR (multi-sujeito);
        # 1 imagem -> IC-LoRA Ingredients (folha de referência única).
        kind = "msr" if len(images) > 1 else "iclora"
    elif images:
        kind = "i2v"
    else:
        kind = "t2v"
    inj = _TEMPLATE_INJECT[kind]

    # ===== MSR: N imagens separadas -> vídeo (fluxo próprio, nós distintos) =====
    if kind == "msr":
        return _build_msr(job_input, inj, images, prompt)
    # ==========================================================================
    path = os.path.join(WORKFLOWS_DIR, inj["file"])
    with open(path) as f:
        wf = json.load(f)

    if prompt and inj["positive"] in wf:
        wf[inj["positive"]]["inputs"]["text"] = prompt
    neg = job_input.get("negative_prompt")
    if neg and inj["negative"] in wf:
        wf[inj["negative"]]["inputs"]["text"] = neg
    # IC-LoRA: a resolução do vídeo deriva da FOLHA de referência PRESERVANDO o aspecto
    # (como no workflow proven; sem isso, quadrado forçado distorce). Resize p/ ~área alvo
    # (múltiplo de 32) e casa o latente (3059) com essas dims.
    if kind == "iclora" and images and inj.get("empty_latent") in wf:
        images[0]["image"], dims = _fit_area(images[0]["image"])
        if dims:
            wf[inj["empty_latent"]]["inputs"]["width"] = dims[0]
            wf[inj["empty_latent"]]["inputs"]["height"] = dims[1]
            print(f"worker-ltx-video - iclora: ref sheet {dims[0]}x{dims[1]} (aspecto preservado)")

    # Aspect_ratio do node -> resolução (t2v: EmptyImage; i2v: resize cover da imagem).
    aspect = (job_input.get("aspect_ratio") or "").strip()
    wh = _ASPECT_WH.get(aspect)
    if wh:
        w, h = wh
        if kind == "t2v" and inj.get("empty_image") and inj["empty_image"] in wf:
            wf[inj["empty_image"]]["inputs"]["width"] = w
            wf[inj["empty_image"]]["inputs"]["height"] = h
            print(f"worker-ltx-video - aspect {aspect} -> t2v {w}x{h}")
        elif kind == "i2v" and images:
            images[0]["image"] = _resize_cover(images[0]["image"], w, h)
            print(f"worker-ltx-video - aspect {aspect} -> i2v resize cover {w}x{h}")
    elif kind == "i2v" and images:
        # Sem aspect_ratio: a resolução deriva da imagem — limita o lado maior p/ não
        # estourar timeout/VRAM com imagem grande (mantém o aspecto da imagem).
        images[0]["image"] = _cap_longside(images[0]["image"], _LTX_MAX_LONGSIDE)

    if kind == "i2v" and inj["load_image"] in wf and images:
        wf[inj["load_image"]]["inputs"]["image"] = images[0]["name"]

    # Duração do node -> nº de frames (LTX = 8n+1). `duration`/`duration_seconds` (s) tem
    # prioridade; `num_frames` é o override direto. Sem nenhum, mantém o default do template.
    fps = inj.get("fps") or 24
    frames = None
    if job_input.get("num_frames"):
        frames = int(job_input["num_frames"])
    elif job_input.get("duration") or job_input.get("duration_seconds"):
        secs = float(job_input.get("duration") or job_input.get("duration_seconds"))
        frames = round(secs * fps)
    if frames and inj.get("length") and inj["length"] in wf:
        snapped = _snap_frames(frames, _LTX_MAX_FRAMES)
        wf[inj["length"]]["inputs"]["value"] = snapped
        print(f"worker-ltx-video - duração: pedido={frames}f -> aplicado={snapped}f (~{snapped/fps:.1f}s @ {fps}fps)")

    # Movimento: img_compression maior solta o modelo da imagem de entrada (i2v).
    comp = job_input.get("img_compression")
    if comp and inj.get("preprocess") and inj["preprocess"] in wf:
        wf[inj["preprocess"]]["inputs"]["img_compression"] = int(comp)

    print(f"worker-ltx-video - modo-prompt: {kind} | imagens={len(images)} | prompt={prompt[:60]!r}")
    return wf, images


def validate_input(job_input):
    """Valida input do job."""
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    workflow = job_input.get("workflow")
    # Modo-prompt: sem `workflow` mas com `prompt` (ou imagens) → constrói o workflow LTX.
    if workflow is None and (job_input.get("prompt") or job_input.get("images") or job_input.get("image_urls") or job_input.get("reference_images")):
        try:
            workflow, built_images = build_workflow_from_prompt(job_input)
            if job_input.get("images") is None:
                job_input["images"] = built_images
            else:
                job_input["images"] = built_images
        except Exception as e:
            return None, f"Falha ao montar workflow do prompt: {e}"

    if workflow is None:
        return None, "Missing 'workflow' parameter"
    if not _has_output_node(workflow):
        return None, "Workflow sem output node reconhecido (ex: VHS_VideoCombine/SaveVideo)"

    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            "name" in image and "image" in image for image in images
        ):
            return None, "'images' must be a list of objects with 'name' and 'image' keys"

    # Áudio p/ talking-avatar (InfiniteTalk): lista [{name, audio(base64)}] ou single.
    audio = job_input.get("audio")
    if audio is not None:
        if isinstance(audio, dict):
            audio = [audio]
        if not isinstance(audio, list) or not all(
            isinstance(a, dict) and "name" in a and "audio" in a for a in audio
        ):
            return None, "'audio' must be a list of objects with 'name' and 'audio' keys"

    comfy_org_api_key = job_input.get("comfy_org_api_key")
    return {"workflow": workflow, "images": images, "audio": audio, "comfy_org_api_key": comfy_org_api_key}, None


def check_server(url, retries=500, delay=50):
    """Verifica se o servidor ComfyUI está acessível."""
    print(f"worker-ltx-video - Verificando API em {url}...")
    for i in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"worker-ltx-video - API acessível")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException:
            pass
        time.sleep(delay / 1000)

    print(f"worker-ltx-video - Falha ao conectar em {url} após {retries} tentativas.")
    return False


def upload_images(images):
    """Grava imagens decodificadas em /comfyui/input e usa upload HTTP apenas como fallback."""
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []
    print(f"worker-ltx-video - Uploading {len(images)} imagem(ns)...")

    for image in images:
        blob = None
        try:
            name = image["name"]
            image_data_uri = image["image"]
            if "," in image_data_uri:
                base64_data = image_data_uri.split(",", 1)[1].strip()
            else:
                base64_data = image_data_uri.strip()

            blob = base64.b64decode(base64_data)
            print(
                "worker-ltx-video - Decoded image summary:",
                json.dumps({"name": name, **_summarize_bytes(blob)}, ensure_ascii=False),
            )

            input_dir = "/comfyui/input"
            os.makedirs(input_dir, exist_ok=True)
            dst_path = os.path.join(input_dir, name)

            if Image is None:
                with open(dst_path, "wb") as fh:
                    fh.write(blob)
            else:
                with Image.open(BytesIO(blob)) as img:
                    img.load()
                    normalized = img
                    if img.mode not in ("RGB", "RGBA"):
                        normalized = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                    with BytesIO() as buf:
                        normalized.save(buf, format="PNG")
                        normalized_blob = buf.getvalue()
                with open(dst_path, "wb") as fh:
                    fh.write(normalized_blob)

            probe = _probe_image_path(dst_path)
            print(
                "worker-ltx-video - Saved image probe:",
                json.dumps({"name": name, "probe": probe}, ensure_ascii=False),
            )
            if probe.get("pillow_error"):
                raise ValueError(f"imagem salva inválida: {probe['pillow_error']}")

            file_size = os.path.getsize(dst_path)
            responses.append(f"Saved OK: {name} ({file_size} bytes)")
        except Exception as e:
            # Loga o erro ANTES de tentar o fallback
            print(f"worker-ltx-video - Primary save FAILED ({type(e).__name__}): {e}")

            # Fallback para endpoint /upload/image
            if blob is None:
                upload_errors.append(f"base64 decode falhou para {image.get('name', '?')}: {e}")
                continue
            try:
                detected_format = _detect_image_format(blob)
                if detected_format == "jpeg":
                    fallback_mime = "image/jpeg"
                elif detected_format == "webp":
                    fallback_mime = "image/webp"
                elif detected_format == "gif":
                    fallback_mime = "image/gif"
                else:
                    fallback_mime = "image/png"

                files = {
                    "image": (image.get("name", "input.png"), BytesIO(blob), fallback_mime),
                    "overwrite": (None, "true"),
                }
                response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=30)
                response.raise_for_status()
                fallback_resp = response.json() if response.content else {}

                # Verifica se o arquivo realmente ficou acessível após o upload
                dst_path = os.path.join("/comfyui/input", image.get("name", "input.png"))
                if os.path.exists(dst_path):
                    probe = _probe_image_path(dst_path)
                    print(
                        "worker-ltx-video - Fallback image probe:",
                        json.dumps({"name": image.get("name"), "probe": probe, "response": fallback_resp}, ensure_ascii=False),
                    )
                else:
                    print(f"worker-ltx-video - AVISO: fallback upload OK mas arquivo nao encontrado em {dst_path}. Resp: {fallback_resp}")

                responses.append(f"Upload OK (fallback): {image.get('name', 'unknown')}")
                print(f"worker-ltx-video - Upload OK (fallback): {image.get('name', 'unknown')}")
            except Exception as fallback_error:
                error_msg = (
                    f"Erro no upload de {image.get('name', 'unknown')}: primary={e}; "
                    f"fallback={fallback_error}"
                )
                print(f"worker-ltx-video - {error_msg}")
                upload_errors.append(error_msg)

    if upload_errors:
        return {"status": "error", "message": "Algumas imagens falharam", "details": upload_errors}
    return {"status": "success", "message": "Todas as imagens enviadas", "details": responses}


def upload_audio(audio_files):
    """Grava áudios decodificados em /comfyui/input (LoadAudio lê de lá)."""
    if not audio_files:
        return {"status": "success", "message": "No audio to upload", "details": []}

    responses = []
    upload_errors = []
    input_dir = "/comfyui/input"
    os.makedirs(input_dir, exist_ok=True)
    print(f"worker-infinitetalk - Uploading {len(audio_files)} áudio(s)...")

    for item in audio_files:
        try:
            name = item["name"]
            data_uri = item["audio"]
            base64_data = data_uri.split(",", 1)[1].strip() if "," in data_uri else data_uri.strip()
            blob = base64.b64decode(base64_data)
            dst_path = os.path.join(input_dir, name)
            with open(dst_path, "wb") as fh:
                fh.write(blob)
            size = os.path.getsize(dst_path)
            print(f"worker-infinitetalk - Áudio salvo: {name} ({size} bytes)")
            responses.append(f"Saved OK: {name} ({size} bytes)")
        except Exception as e:
            msg = f"Erro no upload de áudio {item.get('name', '?')}: {e}"
            print(f"worker-infinitetalk - {msg}")
            upload_errors.append(msg)

    if upload_errors:
        return {"status": "error", "message": "Algum áudio falhou", "details": upload_errors}
    return {"status": "success", "message": "Áudios enviados", "details": responses}


def queue_workflow(workflow, client_id, comfy_org_api_key=None):
    """Enfileira workflow no ComfyUI."""
    payload = {"prompt": workflow, "client_id": client_id}

    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30)

    if response.status_code == 400:
        print(f"worker-ltx-video - ComfyUI 400: {response.text}")
        try:
            error_data = response.json()
            error_message = "Workflow validation failed"
            error_details = []

            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                else:
                    error_message = str(error_info)

            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(f"Node {node_id} ({error_type}): {error_msg}")
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            if error_details:
                raise ValueError(f"{error_message}:\n" + "\n".join(f"  {d}" for d in error_details))
            else:
                raise ValueError(f"{error_message}. Raw: {response.text}")
        except (json.JSONDecodeError, KeyError):
            raise ValueError(f"ComfyUI validation failed: {response.text}")

    response.raise_for_status()
    return response.json()


def get_history(prompt_id):
    """Recupera histórico do prompt."""
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_file_data(filename, subfolder, file_type):
    """Busca dados de arquivo do endpoint /view do ComfyUI."""
    data = {"filename": filename, "subfolder": subfolder, "type": file_type}
    url_values = urllib.parse.urlencode(data)
    try:
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=120)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"worker-ltx-video - Erro ao buscar {filename}: {e}")
        return None


def handler(job):
    """Handler principal para jobs de geração de vídeo com LTX."""
    if is_network_volume_debug_enabled():
        run_network_volume_diagnostics()

    job_input = job["input"]
    job_id = job["id"]

    # Diagnóstico: reproduz o import de um custom node num subprocess e captura a exceção
    # EXATA (o /object_info só diz "ausente"; o log de startup é dessincronizado). Uso:
    # {"input":{"import_probe":"ComfyUI-WanVideoWrapper"}}  (nome da pasta em custom_nodes)
    if job_input.get("import_probe"):
        import subprocess as _sp
        pkg = job_input.get("import_probe")
        code = (
            "import sys, traceback, importlib\n"
            "sys.path.insert(0, '/comfyui')\n"
            "sys.path.insert(0, '/comfyui/custom_nodes')\n"
            "try:\n"
            "    importlib.import_module(%r)\n"
            "    print('IMPORT_OK')\n"
            "except Exception:\n"
            "    traceback.print_exc()\n"
        ) % pkg
        try:
            r = _sp.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
            out = (r.stdout or "") + (r.stderr or "")
        except Exception as e:
            out = f"(falha ao rodar probe: {e})"
        return {"import_probe": pkg, "result": out[-6000:]}

    # Introspection: devolve o /object_info do ComfyUI (schemas exatos dos nodes).
    # Usado p/ converter workflow UI->API sem adivinhar ordem de widgets.
    if job_input.get("object_info"):
        if not check_server(
            f"http://{COMFY_HOST}/",
            COMFY_API_AVAILABLE_MAX_RETRIES,
            COMFY_API_AVAILABLE_INTERVAL_MS,
        ):
            return {"error": "ComfyUI inacessível para object_info"}
        try:
            only = job_input.get("object_info")
            url = f"http://{COMFY_HOST}/object_info"
            if isinstance(only, str) and only not in ("1", "true", "all"):
                url = f"{url}/{urllib.parse.quote(only)}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return {"object_info": resp.json()}
        except Exception as e:
            return {"error": f"Falha ao obter object_info: {e}"}

    # Diagnóstico: log de startup do ComfyUI (achar import falho de custom node).
    # Espera o ComfyUI subir (check_server) p/ garantir que os imports já foram logados,
    # e grepa o ARQUIVO INTEIRO com contexto (não só o tail) quando há needle.
    if job_input.get("logtail"):
        check_server(
            f"http://{COMFY_HOST}/",
            COMFY_API_AVAILABLE_MAX_RETRIES,
            COMFY_API_AVAILABLE_INTERVAL_MS,
        )
        needle = job_input.get("logtail")
        if isinstance(needle, str) and needle not in ("1", "true", "all"):
            hits = _grep_comfy_log(needle)
            return {"logtail": hits[-300:]}
        return {"logtail": _read_comfy_log_tail(max_lines=600)[-250:]}

    validated_data, error_message = validate_input(job_input)
    if error_message:
        diagnostics = _build_runtime_diagnostics(error_message)
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-ltx-video - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": error_message, "diagnostics": diagnostics}

    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")
    input_audio = validated_data.get("audio")
    _log_job_diagnostics(job_id, job_input, workflow, input_images)

    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        message = f"ComfyUI ({COMFY_HOST}) inacessível após múltiplas tentativas."
        startup_tail = _read_comfy_log_tail()
        diagnostics = _build_runtime_diagnostics([message] + startup_tail)
        if startup_tail:
            print("worker-ltx-video - Comfy startup log tail:\n" + "\n".join(startup_tail[-40:]))
        return {
            "error": message,
            "details": {
                "comfy_startup_log_tail": startup_tail[-120:],
                "wait_ms": COMFY_API_AVAILABLE_MAX_RETRIES * COMFY_API_AVAILABLE_INTERVAL_MS,
            },
            "diagnostics": diagnostics,
        }

    if input_images:
        upload_result = upload_images(input_images)
        if upload_result["status"] == "error":
            diagnostics = _build_runtime_diagnostics(upload_result.get("details"))
            return {
                "error": "Falha no upload de imagens",
                "details": upload_result["details"],
                "diagnostics": diagnostics,
            }
        preflight_checks = _preflight_loadimage_inputs(workflow)
        if preflight_checks:
            print(
                "worker-ltx-video - LoadImage preflight:",
                json.dumps(preflight_checks, ensure_ascii=False),
            )
            invalid_checks = [
                check for check in preflight_checks
                if not check.get("probe", {}).get("exists")
                or check.get("probe", {}).get("pillow_error")
            ]
            if invalid_checks:
                diagnostics = _build_runtime_diagnostics(json.dumps(invalid_checks, ensure_ascii=False))
                return {
                    "error": "Imagem de entrada inválida antes do queue_workflow",
                    "details": invalid_checks,
                    "diagnostics": diagnostics,
                }

    if input_audio:
        audio_result = upload_audio(input_audio)
        if audio_result["status"] == "error":
            return {
                "error": "Falha no upload de áudio",
                "details": audio_result["details"],
            }

    gemma_preflight = _preflight_gemma_loader(workflow)
    if gemma_preflight:
        print(
            "worker-ltx-video - Gemma loader preflight:",
            json.dumps(gemma_preflight, ensure_ascii=False),
        )

    ws = None
    client_id = str(uuid.uuid4())
    prompt_id = None
    output_data = []
    video_data = []
    audio_data = []
    errors = []
    history_output_summary = []
    workflow_nodes = _build_workflow_node_lookup(workflow)

    try:
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-ltx-video - Conectando ao websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print(f"worker-ltx-video - Websocket conectado")

        try:
            queued_workflow = queue_workflow(
                workflow, client_id,
                comfy_org_api_key=validated_data.get("comfy_org_api_key"),
            )
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"prompt_id ausente na resposta: {queued_workflow}")
            print(f"worker-ltx-video - Workflow enfileirado: {prompt_id}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Erro ao enfileirar workflow: {e}")

        print(f"worker-ltx-video - Aguardando execução ({prompt_id})...")
        execution_done = False
        last_event_at = time.time()
        last_queue_remaining = None
        current_node_id = None
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    message_type = message.get("type")
                    data = message.get("data", {})
                    if message_type == "status":
                        status_data = message.get("data", {}).get("status", {})
                        queue_remaining = status_data.get("exec_info", {}).get("queue_remaining", "N/A")
                        if queue_remaining != last_queue_remaining:
                            print(f"worker-ltx-video - Queue: {queue_remaining} restantes")
                            last_queue_remaining = queue_remaining
                        last_event_at = time.time()
                    elif message_type == "execution_start":
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event("Execution start", {"prompt_id": prompt_id})
                            last_event_at = time.time()
                    elif message_type == "execution_cached":
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event(
                                "Execution cached",
                                {
                                    "prompt_id": prompt_id,
                                    "nodes": data.get("nodes", []),
                                },
                            )
                            last_event_at = time.time()
                    elif message_type == "executing":
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            print(f"worker-ltx-video - Execução finalizada: {prompt_id}")
                            execution_done = True
                            break
                        if data.get("prompt_id") == prompt_id:
                            current_node_id = str(data.get("node")) if data.get("node") is not None else None
                            current_node_class = workflow_nodes.get(current_node_id, "unknown") if current_node_id else None
                            _log_ws_event(
                                "Executing node",
                                {
                                    "prompt_id": prompt_id,
                                    "node_id": current_node_id,
                                    "class_type": current_node_class,
                                    "idle_timeout_s": _get_idle_timeout_for_node_class(current_node_class),
                                },
                            )
                            last_event_at = time.time()
                    elif message_type == "progress":
                        value = data.get("value")
                        max_value = data.get("max")
                        progress_payload = {
                            "prompt_id": prompt_id,
                            "node_id": current_node_id,
                            "class_type": workflow_nodes.get(current_node_id, "unknown") if current_node_id else None,
                            "value": value,
                            "max": max_value,
                        }
                        _log_ws_event("Progress", progress_payload)
                        last_event_at = time.time()
                    elif message_type == "executed":
                        if data.get("prompt_id") == prompt_id:
                            node_id = str(data.get("node")) if data.get("node") is not None else None
                            output = data.get("output")
                            _log_ws_event(
                                "Executed node",
                                {
                                    "prompt_id": prompt_id,
                                    "node_id": node_id,
                                    "class_type": workflow_nodes.get(node_id, "unknown") if node_id else None,
                                    "output_keys": _safe_dict_keys(output),
                                },
                            )
                            last_event_at = time.time()
                    elif message_type == "execution_success":
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event("Execution success", {"prompt_id": prompt_id})
                            last_event_at = time.time()
                    elif message_type == "execution_interrupted":
                        if data.get("prompt_id") == prompt_id:
                            errors.append("Execution interrupted pelo ComfyUI")
                            _log_ws_event("Execution interrupted", {"prompt_id": prompt_id})
                            break
                    elif message_type == "execution_error":
                        if data.get("prompt_id") == prompt_id:
                            error_details = f"Node: {data.get('node_type')}, ID: {data.get('node_id')}, Msg: {data.get('exception_message')}"
                            print(f"worker-ltx-video - Erro de execução: {error_details}")
                            errors.append(f"Execution error: {error_details}")
                            break
                    else:
                        if data.get("prompt_id") == prompt_id:
                            _log_ws_event("WS event", {"type": message_type, "data_keys": _safe_dict_keys(data)})
                            last_event_at = time.time()
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                idle_for = int(time.time() - last_event_at)
                current_node_class = workflow_nodes.get(current_node_id, "unknown") if current_node_id else None
                idle_limit = _get_idle_timeout_for_node_class(current_node_class)
                if idle_for >= idle_limit:
                    current_node_class = workflow_nodes.get(current_node_id, "unknown") if current_node_id else None
                    raise ValueError(
                        f"Sem eventos do websocket por {idle_for}s após enqueue do prompt {prompt_id}. "
                        f"Último node conhecido: {current_node_id or 'none'} ({current_node_class or 'unknown'}). "
                        f"Limite aplicado: {idle_limit}s."
                    )
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    ws = _attempt_websocket_reconnect(
                        ws_url, WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S, closed_err,
                    )
                    continue
                except websocket.WebSocketConnectionClosedException as reconn_err:
                    raise reconn_err
            except json.JSONDecodeError:
                print(f"worker-ltx-video - JSON inválido no websocket.")

        if not execution_done and not errors:
            raise ValueError("Loop de monitoramento terminou sem confirmação.")

        print(f"worker-ltx-video - Buscando histórico de {prompt_id}...")
        history = get_history(prompt_id)

        if prompt_id not in history:
            error_msg = f"Prompt {prompt_id} não encontrado no histórico."
            if not errors:
                diagnostics = _build_runtime_diagnostics(error_msg)
                return {"error": error_msg, "diagnostics": diagnostics}
            errors.append(error_msg)
            diagnostics = _build_runtime_diagnostics(errors)
            return {"error": "Job falhou", "details": errors, "diagnostics": diagnostics}

        outputs = history.get(prompt_id, {}).get("outputs", {})
        print(f"worker-ltx-video - Processando {len(outputs)} nós de saída...")

        for node_id, node_output in outputs.items():
            if not isinstance(node_output, dict):
                history_output_summary.append(
                    {"node_id": str(node_id), "keys": [], "warning": "node_output not dict"}
                )
                continue

            if isinstance(node_output, dict):
                history_output_summary.append(
                    {"node_id": str(node_id), "keys": sorted(list(node_output.keys()))}
                )

            def normalize_output_entries(raw_value, label):
                entries = []
                if isinstance(raw_value, list):
                    for idx, item in enumerate(raw_value):
                        if isinstance(item, dict):
                            entries.append(item)
                        else:
                            print(
                                "worker-ltx-video - Ignorando item de saída não estruturado:",
                                json.dumps(
                                    {
                                        "label": label,
                                        "index": idx,
                                        "python_type": type(item).__name__,
                                        "value": item,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                elif isinstance(raw_value, dict):
                    entries.append(raw_value)
                elif raw_value is not None:
                    print(
                        "worker-ltx-video - Ignorando saída não estruturada:",
                        json.dumps(
                            {
                                "label": label,
                                "python_type": type(raw_value).__name__,
                                "value": raw_value,
                            },
                            ensure_ascii=False,
                        ),
                    )
                return entries

            # Imagens
            if "images" in node_output:
                for image_info in normalize_output_entries(node_output["images"], "images"):
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    img_type = image_info.get("type")

                    if img_type == "temp":
                        continue
                    if not filename:
                        continue

                    image_bytes = get_file_data(filename, subfolder, img_type)
                    if image_bytes:
                        # SaveVideo expõe o mp4 sob "images". MSR: corta a folha de
                        # referência que sangra nos 1ºs frames (trim frame-exato).
                        if filename.lower().endswith((".mp4", ".webm", ".mov")) and (
                            isinstance(workflow, dict)
                            and any(
                                isinstance(n, dict) and n.get("class_type") == "LiconMSR"
                                for n in workflow.values()
                            )
                        ):
                            image_bytes = _trim_leading_frames(image_bytes, filename, _MSR_TRIM_FRAMES)
                        if os.environ.get("BUCKET_ENDPOINT_URL"):
                            try:
                                file_ext = os.path.splitext(filename)[1] or ".png"
                                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                                    tmp.write(image_bytes)
                                    tmp_path = tmp.name
                                s3_url = rp_upload.upload_image(job_id, tmp_path)
                                os.remove(tmp_path)
                                output_data.append({"filename": filename, "type": "s3_url", "data": s3_url})
                            except Exception as e:
                                errors.append(f"Erro S3 upload {filename}: {e}")
                        else:
                            b64 = base64.b64encode(image_bytes).decode("utf-8")
                            output_data.append({"filename": filename, "type": "base64", "data": b64})
                    else:
                        errors.append(f"Falha ao ler imagem {filename} via /view")

            # Vídeos (VHS usa "gifs"; SaveVideo pode expor "videos" ou "animated")
            video_entries = []
            if any(key in node_output for key in ("gifs", "videos", "animated", "video")):
                print(
                    "worker-ltx-video - Video output raw summary:",
                    json.dumps(
                        {
                            "node_id": str(node_id),
                            "gifs_type": type(node_output.get("gifs")).__name__ if "gifs" in node_output else None,
                            "videos_type": type(node_output.get("videos")).__name__ if "videos" in node_output else None,
                            "animated_type": type(node_output.get("animated")).__name__ if "animated" in node_output else None,
                            "video_type": type(node_output.get("video")).__name__ if "video" in node_output else None,
                            "animated_preview": node_output.get("animated"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
            video_entries.extend(normalize_output_entries(node_output.get("gifs"), "gifs"))
            video_entries.extend(normalize_output_entries(node_output.get("videos"), "videos"))
            video_entries.extend(normalize_output_entries(node_output.get("animated"), "animated"))
            video_entries.extend(normalize_output_entries(node_output.get("video"), "video"))

            if video_entries:
                for vid_info in video_entries:
                    filename = vid_info.get("filename")
                    subfolder = vid_info.get("subfolder", "")
                    vid_type = vid_info.get("type", "output")

                    if not filename:
                        continue

                    vid_bytes = get_file_data(filename, subfolder, vid_type)
                    if vid_bytes:
                        vid_bytes, filename = convert_video_to_mp4(vid_bytes, filename)
                        # MSR: corta a folha de referência que sangra nos 1ºs frames.
                        if isinstance(workflow, dict) and any(
                            isinstance(n, dict) and n.get("class_type") == "LiconMSR"
                            for n in workflow.values()
                        ):
                            vid_bytes = _trim_leading_frames(vid_bytes, filename, _MSR_TRIM_FRAMES)
                        try:
                            uploaded_url = upload_binary_artifact(job_id, vid_bytes, filename, ".mp4")
                            print(
                                "worker-ltx-video - Video upload ok:",
                                json.dumps(
                                    {
                                        "filename": filename,
                                        "bytes": len(vid_bytes),
                                        "url": uploaded_url,
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                            video_data.append({"filename": filename, "type": "s3_url", "data": uploaded_url})
                        except Exception as e:
                            print(f"worker-ltx-video - Upload remoto de vídeo falhou ({filename}): {e}. Fallback base64.")
                            if len(vid_bytes) > MAX_INLINE_VIDEO_BYTES:
                                errors.append(
                                    f"Vídeo {filename} excede limite inline ({len(vid_bytes)} bytes > {MAX_INLINE_VIDEO_BYTES}) "
                                    "e upload remoto falhou"
                                )
                            else:
                                b64 = base64.b64encode(vid_bytes).decode("utf-8")
                                video_data.append({"filename": filename, "type": "base64", "data": b64})
                    else:
                        errors.append(f"Falha ao ler vídeo {filename} via /view")

            # Áudio (LTX-2 pode gerar áudio sincronizado)
            if "audio" in node_output:
                for audio_info in normalize_output_entries(node_output["audio"], "audio"):
                    filename = audio_info.get("filename")
                    subfolder = audio_info.get("subfolder", "")
                    audio_type = audio_info.get("type", "temp")

                    if not filename:
                        continue

                    audio_bytes = get_file_data(filename, subfolder, audio_type)
                    if audio_bytes:
                        b64 = base64.b64encode(audio_bytes).decode("utf-8")
                        audio_data.append({"filename": filename, "type": "base64", "data": b64})
                    else:
                        errors.append(f"Falha ao ler áudio {filename} via /view")

    except websocket.WebSocketException as e:
        print(f"worker-ltx-video - WebSocket Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-ltx-video - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": f"WebSocket error: {e}", "diagnostics": diagnostics}
    except requests.RequestException as e:
        print(f"worker-ltx-video - HTTP Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-ltx-video - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": f"HTTP error: {e}", "diagnostics": diagnostics}
    except ValueError as e:
        print(f"worker-ltx-video - Value Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-ltx-video - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": str(e), "diagnostics": diagnostics}
    except Exception as e:
        print(f"worker-ltx-video - Unexpected Error: {e}")
        print(traceback.format_exc())
        diagnostics = _build_runtime_diagnostics(str(e))
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-ltx-video - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": f"Erro inesperado: {e}", "diagnostics": diagnostics}
    finally:
        if ws and ws.connected:
            ws.close()

    final_result = {}

    if output_data:
        final_result["images"] = output_data

    if video_data:
        final_result["video"] = video_data[0]["data"]
        final_result["video_filename"] = video_data[0]["filename"]
        print(
            "worker-ltx-video - Final video selected:",
            json.dumps(
                {
                    "filename": video_data[0]["filename"],
                    "type": video_data[0]["type"],
                    "url": video_data[0]["data"] if video_data[0]["type"] == "s3_url" else None,
                },
                ensure_ascii=False,
            ),
        )
        if len(video_data) > 1:
            final_result["videos"] = video_data

    if audio_data:
        final_result["audio"] = audio_data[0]["data"]
        final_result["audio_filename"] = audio_data[0]["filename"]

    if errors:
        final_result["errors"] = errors
        diagnostics = _build_runtime_diagnostics(errors)
        if diagnostics:
            final_result["diagnostics"] = diagnostics
            if diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
                print("worker-ltx-video - Runtime diagnostics:", json.dumps(diagnostics))

    has_output = output_data or video_data or audio_data
    if not has_output and errors:
        diagnostics = _build_runtime_diagnostics(errors)
        if diagnostics and diagnostics.get("category") in {"GPU_OOM", "GPU_RUNTIME"}:
            print("worker-ltx-video - Runtime diagnostics:", json.dumps(diagnostics))
        return {"error": "Job falhou sem output", "details": errors, "diagnostics": diagnostics}
    elif not has_output and not errors:
        diagnostics = _build_runtime_diagnostics(history_output_summary)
        return {
            "error": "Job concluído sem mídia no histórico do ComfyUI",
            "details": history_output_summary,
            "diagnostics": diagnostics,
        }

    print(f"worker-ltx-video - Job concluído: {len(output_data)} img, {len(video_data)} vid, {len(audio_data)} audio")
    return final_result


if __name__ == "__main__":
    print("worker-ltx-video - Starting handler...")
    runpod.serverless.start({"handler": handler})
