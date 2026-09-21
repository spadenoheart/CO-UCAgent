#!/usr/bin/env python3
import os
import sys
import logging
# 初始化 logger，避免后续使用时未定义导致 NameError
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("embedding_server")
logger.setLevel(logging.INFO)
# ========== 彻底禁用 amdsmi ==========
# 方法1: 设置环境变量
os.environ['PYTORCH_ROCM_AVOID_AMDSMI'] = '1'

# 方法2: 在导入任何模块之前，猴子补丁 sys.modules
class FakeAmdsmi:
    """假的 amdsmi 模块，防止导入错误"""
    class AmdSmiException(Exception):
        pass
    
    @staticmethod
    def amdsmi_init():
        raise FakeAmdsmi.AmdSmiException("amdsmi disabled")
    
    @staticmethod
    def amdsmi_get_device_count():
        return 0

# 在 sys.modules 中插入假模块
sys.modules['amdsmi'] = FakeAmdsmi

# ========== 设置 ROCm 环境 ==========
rocm_paths = [
    '/opt/rocm-6.1.0/lib',
    '/opt/rocm/lib',
    '/usr/local/rocm/lib'
]

for path in rocm_paths:
    if os.path.exists(path):
        os.environ['LD_LIBRARY_PATH'] = f"{path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        break

# 对于 RX 7800 XT (gfx1101)
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

print("=== 环境设置完成 ===")
print(f"PYTORCH_ROCM_AVOID_AMDSMI: {os.environ.get('PYTORCH_ROCM_AVOID_AMDSMI')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')[:100]}...")

# ========== 导入并测试 torch ==========
try:
    import torch
    print(f"✅ PyTorch 导入成功! 版本: {torch.__version__}")
    
    # 手动初始化 CUDA，绕过设备计数
    def safe_cuda_init():
        """安全的 CUDA 初始化，绕过有问题的代码"""
        try:
            # 直接设置 CUDA 为可用状态
            torch._C._cuda_init()
            return True
        except:
            return False
    
    # 尝试初始化 CUDA
    if safe_cuda_init():
        print("✅ CUDA 初始化成功")
    else:
        print("⚠️  CUDA 初始化失败，使用回退方案")
    
    # 检查 CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    
    if cuda_available:
        try:
            # 直接测试 GPU，不调用 device_count()
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print(f"✅ GPU 测试成功! 张量设备: {x.device}")
        except Exception as e:
            print(f"❌ GPU 测试失败: {e}")
            cuda_available = False
    
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    sys.exit(1)

# ========== 导入其他模块 ==========
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Optional
import uvicorn

app = FastAPI(title="Qwen3 Embedding API")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置模型路径与 mock 模式
model_path = os.environ.get("EMBEDDING_MODEL_PATH", "<HOME>/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B")
mock_mode = os.environ.get("EMBEDDING_SERVER_MOCK", "0") == "1"
logger.info(f"准备加载模型: {model_path} (mock_mode={mock_mode})")

# 确定使用设备
device = "cuda" if cuda_available else "cpu"
logger.info(f"使用设备: {device}")

# 加载 tokenizer（mock 模式下跳过）
tokenizer = None
if not mock_mode:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("✅ Tokenizer 加载成功")
    except Exception as e:
        logger.error(f"❌ Tokenizer 加载失败: {e}")
        tokenizer = None

# 加载模型（mock 模式下跳过）
model = None
if not mock_mode and tokenizer is not None:
    try:
        if device == "cuda":
            logger.info("使用 CUDA 模式，加载半精度模型...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=None
            )
            model = model.cuda()
        else:
            logger.info("使用 CPU 模式，加载全精度模型...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map=None
            )
            model = model.eval()

        logger.info("✅ 模型加载完成!")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        # 尝试回退到 CPU
        try:
            logger.info("尝试回退方案：在 CPU 上加载模型...")
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map=None
            )
            if device == "cuda":
                model = model.cuda()
            else:
                model = model.eval()
            logger.info("✅ 回退方案成功!")
        except Exception as e2:
            logger.error(f"❌ 回退方案也失败: {e2}")

# mock 模式下的简单 embedding 生成器（用于调试）
if mock_mode:
    import hashlib
    def mock_get_embedding(text: str, dim: int = 8):
        h = hashlib.sha256(text.encode('utf-8')).digest()
        vec = []
        for i in range(dim):
            # 取两个字节，转换为 0..1 的浮点数
            b0 = h[(i*2) % len(h)]
            b1 = h[(i*2+1) % len(h)]
            val = ((b0 << 8) + b1) % 1000 / 1000.0
            vec.append(float(val))
        return vec

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = "Qwen3-Embedding-0.6B"
    encoding_format: Optional[str] = "float"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict



@app.post("/v1/embeddings")
async def create_embedding(request: Request):
    try:
        # 直接读取原始 JSON，兼容传入 token id 列表的客户端（例如某些 langchain/openai 调用）
        body = await request.json()
        inputs = body.get("input")
        logger.info(f"Received raw input: {type(inputs)}")

        # 必须至少有 tokenizer 或 mock 模式
        if not mock_mode and tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer 未加载，无法处理输入")

        # 规范化 inputs 为字符串列表
        normalized_inputs = []

        # 单个字符串
        if isinstance(inputs, str):
            normalized_inputs = [inputs]

        # 单个 token-id 列表，例如: [11, 22, 33]
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], int):
            # 将 token id 解码为字符串（有 tokenizer 则使用 tokenizer，否则回退为用空格连接数字）
            if tokenizer is not None:
                try:
                    text = tokenizer.decode(inputs, skip_special_tokens=True)
                except Exception:
                    text = " ".join(map(str, inputs))
            else:
                text = " ".join(map(str, inputs))
            normalized_inputs = [text]

        # 列表 - 可能是 list[str] 或 list[list[int]]
        elif isinstance(inputs, list):
            for idx, item in enumerate(inputs):
                if isinstance(item, str):
                    normalized_inputs.append(item)
                elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], int):
                    if tokenizer is not None:
                        try:
                            text = tokenizer.decode(item, skip_special_tokens=True)
                        except Exception:
                            text = " ".join(map(str, item))
                    else:
                        text = " ".join(map(str, item))
                    normalized_inputs.append(text)
                else:
                    # 对于非字符串/非 token-id 的项，尝试强制转换为字符串
                    normalized_inputs.append(str(item))
        else:
            raise HTTPException(status_code=400, detail="Unsupported input type for embeddings")

        logger.info(f"Normalized inputs count={len(normalized_inputs)}")
        
        # 如果是 mock 模式，使用简单生成器返回固定向量，避免依赖真实 tokenizer/model
        if mock_mode:
            embeddings = []
            for idx, text in enumerate(normalized_inputs):
                vec = mock_get_embedding(text)
                embeddings.append({
                    "object": "embedding",
                    "embedding": vec,
                    "index": idx
                })
            return {
                "object": "list",
                "data": embeddings,
                "model": body.get("model", "Qwen3-Embedding-0.6B"),
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            }

        # 获取 embedding
        embeddings = []
        total_tokens = 0

        for text in normalized_inputs:
            # tokenize
            inputs_tokenized = tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=8192
            )
            
            # 计算 token 数量
            token_count = inputs_tokenized['input_ids'].shape[1]
            total_tokens += token_count
            
            # 移动到模型设备
            if device == "cuda":
                inputs_tokenized = {k: v.cuda() for k, v in inputs_tokenized.items()}
            
            # 获取 embedding
            with torch.no_grad():
                outputs = model(**inputs_tokenized)
                # 使用最后一层隐藏状态的平均池化
                attention_mask = inputs_tokenized['attention_mask']
                last_hidden_state = outputs.last_hidden_state
                
                # 应用 attention mask 进行平均池化
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = (sum_embeddings / sum_mask).cpu().numpy()
            
            # 转换为列表
            embedding_list = embedding[0].tolist()
            embeddings.append({
                "object": "embedding",
                "embedding": embedding_list,
                "index": len(embeddings)
            })
        
        return {
            "object": "list",
            "data": embeddings,
            "model": body.get("model", "Qwen3-Embedding-0.6B"),
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")  # 打印错误信息
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/info")
async def get_info():
    """获取服务器信息"""
    import torch
    import platform
    
    return {
        "status": "running",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "python_version": platform.python_version(),
        "system": platform.system(),
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

if __name__ == "__main__":
    print("\n=== 启动服务器 ===")
    print(f"设备: {device}")
    print(f"模型已加载: {model is not None}")
    print(f"Tokenizer 已加载: {tokenizer is not None}")
    print(f"服务器地址: http://0.0.0.0:5000")
    print("接口:")
    print("  POST /v1/embeddings  - 获取文本嵌入")
    print("  GET  /health         - 健康检查")
    print("  GET  /info           - 服务器信息")
    print("\n按 Ctrl+C 停止服务器\n")
    
    uvicorn.run(app, host="0.0.0.0", port=5000)