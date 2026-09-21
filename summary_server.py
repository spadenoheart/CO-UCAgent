#!/usr/bin/env python3
import os
import sys
import json
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("summary_server")
logger.setLevel(logging.INFO)

# ========== ROCm / amdsmi guard ==========
os.environ["PYTORCH_ROCM_AVOID_AMDSMI"] = "1"


class FakeAmdsmi:
    class AmdSmiException(Exception):
        pass

    @staticmethod
    def amdsmi_init():
        raise FakeAmdsmi.AmdSmiException("amdsmi disabled")

    @staticmethod
    def amdsmi_get_device_count():
        return 0


sys.modules["amdsmi"] = FakeAmdsmi

rocm_paths = [
    "/opt/rocm-6.1.0/lib",
    "/opt/rocm/lib",
    "/usr/local/rocm/lib",
]

for path in rocm_paths:
    if os.path.exists(path):
        os.environ["LD_LIBRARY_PATH"] = f"{path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        break

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

print("=== Summary Server 环境设置完成 ===")
print(f"PYTORCH_ROCM_AVOID_AMDSMI: {os.environ.get('PYTORCH_ROCM_AVOID_AMDSMI')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')[:100]}...")

try:
    import torch

    print(f"✅ PyTorch 导入成功! 版本: {torch.__version__}")

    def safe_cuda_init():
        try:
            torch._C._cuda_init()
            return True
        except Exception:
            return False

    if safe_cuda_init():
        print("✅ CUDA 初始化成功")
    else:
        print("⚠️ CUDA 初始化失败，使用回退方案")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    sys.exit(1)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI(title="Qwen3 Summary API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.environ.get(
    "SUMMARY_MODEL_PATH",
    "Qwen/Qwen3.5-4B",
)
mock_mode = os.environ.get("SUMMARY_SERVER_MOCK", "0") == "1"
host = os.environ.get("SUMMARY_SERVER_HOST", "0.0.0.0")
port = int(os.environ.get("SUMMARY_SERVER_PORT", "8001"))
default_max_new_tokens = int(os.environ.get("SUMMARY_MAX_NEW_TOKENS", "2048"))
default_temperature = float(os.environ.get("SUMMARY_TEMPERATURE", "0.2"))
default_top_p = float(os.environ.get("SUMMARY_TOP_P", "0.8"))
served_model_name = os.environ.get("SUMMARY_MODEL_NAME", "Qwen3.5-4B")

logger.info(f"准备加载摘要模型: {model_path} (mock_mode={mock_mode})")
device = "cuda" if cuda_available else "cpu"
logger.info(f"使用设备: {device}")

tokenizer = None
model = None
load_error = None


def _load_model():
    global tokenizer, model
    if mock_mode:
        logger.info("SUMMARY_SERVER_MOCK=1，跳过真实模型加载。")
        return
    if tokenizer is not None and model is not None:
        return
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SUMMARY_MODEL_PATH does not exist: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("✅ Tokenizer 加载成功")

    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"加载摘要模型，dtype={dtype}, device={device}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
    except Exception as exc:
        logger.warning(f"device_map=auto 加载失败: {exc}，回退为普通加载。")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=None,
        )
        if device == "cuda":
            model = model.cuda()
    model.eval()
    logger.info("✅ 摘要模型加载完成")


if not mock_mode:
    try:
        _load_model()
    except Exception as e:
        load_error = e
        tokenizer = None
        model = None
        logger.exception(f"❌ 摘要模型加载失败: {e}")


class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[dict], List[str]]] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "Qwen3.5-4B"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    response_format: Optional[dict] = None


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _normalize_messages(messages: List[Union[ChatMessage, dict]]) -> List[dict]:
    normalized = []
    for msg in messages:
        role = getattr(msg, "role", None) if not isinstance(msg, dict) else msg.get("role")
        content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        name = getattr(msg, "name", None) if not isinstance(msg, dict) else msg.get("name")
        item = {"role": role or "user", "content": _normalize_content(content)}
        if name:
            item["name"] = name
        normalized.append(item)
    return normalized


def _messages_to_text(messages: List[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        name = msg.get("name")
        if name:
            lines.append(f"[{role}:{name}]\n{content}")
        else:
            lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def _extract_json_schema(request_json: dict) -> Optional[dict]:
    response_format = request_json.get("response_format") or {}
    if not isinstance(response_format, dict):
        response_format = {}
    if response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema")
        if isinstance(schema, dict):
            if isinstance(schema.get("schema"), dict):
                return schema["schema"]
            return schema
    tools = request_json.get("tools") or []
    if isinstance(tools, list) and tools:
        fn = tools[0].get("function") or {}
        params = fn.get("parameters")
        if isinstance(params, dict):
            return params
    return None


def _build_json_only_prompt(messages: List[dict], schema: Optional[dict], for_tool_call: bool) -> List[dict]:
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2) if schema else "{}"
    system = {
        "role": "system",
        "content": (
            "You are a structured summarization model.\n"
            "Return ONLY valid JSON. Do not include markdown, code fences, or any extra explanation.\n"
            "If a field is unknown, use an empty string, 0, false, null, or an empty list/object as appropriate.\n"
            f"Target schema:\n{schema_text}"
        ),
    }
    if for_tool_call:
        system["content"] += "\nThe JSON you output will be used as function arguments."
    return [system] + messages


def _apply_chat_template(messages: List[dict]) -> str:
    assert tokenizer is not None
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return _messages_to_text(messages) + "\n\n[ASSISTANT]\n"


def _trim_stop(text: str, stop: Optional[Union[str, List[str]]]) -> str:
    if not stop:
        return text
    stop_list = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)]
    out = text
    for s in stop_list:
        idx = out.find(s)
        if idx >= 0:
            out = out[:idx]
    return out


def _generate_text(messages: List[dict], request_json: dict) -> Dict[str, Any]:
    if mock_mode:
        content = '{"mock": true, "message": "summary server mock mode"}'
        return {
            "content": content,
            "prompt_tokens": 0,
            "completion_tokens": len(content.split()),
            "finish_reason": "stop",
        }
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Summary model/tokenizer not loaded")

    structured_schema = _extract_json_schema(request_json)
    wants_json = structured_schema is not None or (request_json.get("response_format") or {}).get("type") == "json_object"
    tool_mode = bool(request_json.get("tools"))
    if wants_json:
        messages = _build_json_only_prompt(messages, structured_schema, tool_mode)

    prompt = _apply_chat_template(messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[1])
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    max_new_tokens = int(request_json.get("max_tokens") or default_max_new_tokens)
    temperature = float(default_temperature if request_json.get("temperature") is None else request_json["temperature"])
    top_p = float(default_top_p if request_json.get("top_p") is None else request_json["top_p"])
    do_sample = temperature > 0.0

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": max(temperature, 1e-5) if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if request_json.get("seed") is not None:
        torch.manual_seed(int(request_json["seed"]))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(request_json["seed"]))

    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)
    latency_ms = int((time.time() - start) * 1000)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    text = _trim_stop(text, request_json.get("stop"))
    completion_tokens = int(len(generated_ids))
    logger.info(
        "summary generate done: prompt_tokens=%s completion_tokens=%s latency_ms=%s tools=%s response_format=%s",
        prompt_tokens,
        completion_tokens,
        latency_ms,
        bool(request_json.get("tools")),
        (request_json.get("response_format") or {}).get("type"),
    )
    return {
        "content": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": "stop",
        "latency_ms": latency_ms,
    }


def _build_openai_response(request_json: dict, result: Dict[str, Any]) -> dict:
    model_name = request_json.get("model") or served_model_name
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": result["content"],
    }

    tools = request_json.get("tools") or []
    if tools:
        fn = (tools[0] or {}).get("function") or {}
        fn_name = fn.get("name") or "structured_output"
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": result["content"],
                    },
                }
            ],
        }
        finish_reason = "tool_calls"
    else:
        finish_reason = result.get("finish_reason", "stop")

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": int(result.get("prompt_tokens", 0)),
            "completion_tokens": int(result.get("completion_tokens", 0)),
            "total_tokens": int(result.get("prompt_tokens", 0)) + int(result.get("completion_tokens", 0)),
        },
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    try:
        body = await request.json()
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="stream=true is not supported by summary_server.py")
        messages = _normalize_messages(body.get("messages") or [])
        if not messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        result = _generate_text(messages, body)
        return _build_openai_response(body, result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /v1/chat/completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": served_model_name,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
async def health_check():
    healthy = bool(mock_mode or (model is not None and tokenizer is not None))
    return {
        "status": "healthy" if healthy else "unhealthy",
        "model_loaded": model is not None or mock_mode,
        "tokenizer_loaded": tokenizer is not None or mock_mode,
        "load_error": str(load_error) if load_error else "",
    }


@app.get("/info")
async def get_info():
    import platform

    return {
        "status": "running",
        "python_version": platform.python_version(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "model_path": model_path,
        "mock_mode": mock_mode,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "load_error": str(load_error) if load_error else "",
        "port": port,
    }


if __name__ == "__main__":
    print("\n=== 启动 Summary Server ===")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    print(f"模型已加载: {model is not None or mock_mode}")
    if load_error is not None and not mock_mode:
        print(f"模型加载失败: {load_error}")
        print("Summary Server 未启动。请先修复模型加载问题。")
        sys.exit(1)
    print(f"服务器地址: http://{host}:{port}")
    print("接口:")
    print("  POST /v1/chat/completions - OpenAI-compatible chat completions")
    print("  GET  /v1/models            - model list")
    print("  GET  /health               - health check")
    print("  GET  /info                 - server info")
    print("\n建议使用环境:")
    print("  SUMMARY_MODEL_PATH=/path/to/model python summary_server.py")
    print("\n按 Ctrl+C 停止服务器\n")
    uvicorn.run(app, host=host, port=port)
