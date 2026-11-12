import re
import requests
import subprocess
import time


OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"

# Tried in order; a --model override is prepended to this list.
MODEL_PRIORITY = [
    "gemma4:26b",      # MoE (3.8B active): fastest good option on a 24 GB card
    "qwen3.6:27b",     # dense: highest quality that fits in 24 GB
    "qwen2.5:32b",     # previous default, kept as a fallback
]

# The system prompt alone is ~5K tokens, plus history and large tool results.
# Ollama's default num_ctx (4096) would silently truncate the prompt.
NUM_CTX = 16384

# Reasoning models would otherwise emit a thinking block before the JSON,
# which slows every agent step and can confuse the JSON extractor.
THINK = False

REQUEST_TIMEOUT_SECONDS = 600

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def is_server_running():
    try:
        requests.get(OLLAMA_BASE, timeout=3)
        return True
    except Exception:
        return False


def start_server():
    print("Starting Ollama server...")
    subprocess.Popen(["ollama", "serve"])
    time.sleep(5)


def ensure_server():
    if not is_server_running():
        start_server()


def _usable_text(raw):
    """Prefer text outside a think block. If that is empty, keep a block that holds JSON."""
    if not raw:
        return ""
    outside = _THINK_BLOCK.sub("", raw).strip()
    if outside:
        return outside
    for block in _THINK_BLOCK.findall(raw):
        if "{" in block:
            return block.strip()
    return ""


def ask_model(model, prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": THINK,
            "options": {
                "num_ctx": NUM_CTX,
            },
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    text = _usable_text(data.get("response") or "")
    if text:
        return text

    text = _usable_text(data.get("thinking") or "")
    if text:
        print(f"[model] {model} left response empty; using the thinking field")
        return text

    print(
        f"[model] {model} returned nothing "
        f"(done_reason={data.get('done_reason')}, eval_count={data.get('eval_count')})"
    )
    return ""


def ask_with_fallback(prompt, model_override=None):

    ensure_server()

    models = MODEL_PRIORITY

    if model_override:
        models = [model_override] + [m for m in MODEL_PRIORITY if m != model_override]

    for model in models:

        try:
            print(f"Trying model: {model}")

            response = ask_model(model, prompt)

            return response

        except Exception as e:

            print(f"Model failed: {model} ({e})")
            continue

    raise Exception("All models failed")
