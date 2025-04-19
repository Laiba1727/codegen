from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import autopep8
import subprocess
import time
import re
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Code Evaluation API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Environment Setup
CACHE_DIR = Path("./.cache/huggingface")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)

# Hugging Face Token from Railway
HF_TOKEN = os.getenv("HF_API_TOKEN")

# Model Configuration
MODEL_NAME = "Salesforce/codet5-base"
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSeq2SeqLM] = None

# Request Model
class CodeRequest(BaseModel):
    code: str
    language: str = "python"

# Evaluate code
def evaluate_code(user_code: str, lang: str) -> dict:
    try:
        start_time = time.time()
        file_ext = {"python": "py", "java": "java", "cpp": "cpp", "javascript": "js"}.get(lang, "txt")
        filename = f"temp_script.{file_ext}"

        with open(filename, "w") as f:
            f.write(user_code)

        commands = {
            "python": ["python3", filename],
            "java": ["javac", filename, "&&", "java", filename.replace(".java", "")],
            "cpp": ["g++", filename, "-o", "temp_script.out", "&&", "./temp_script.out"],
            "javascript": ["node", filename]
        }

        if lang in commands:
            result = subprocess.run(" ".join(commands[lang]), 
                                    capture_output=True, 
                                    text=True, 
                                    timeout=15, 
                                    shell=True)
            exec_time = time.time() - start_time
            correctness = 1 if result.returncode == 0 else 0
            error_message = None if correctness else result.stderr.strip()
        else:
            return {"status": "error", "message": "Unsupported language", "score": 0}

        readability_score = 20 if len(user_code) < 200 else 10
        efficiency_score = 30 if exec_time < 1 else 10
        security_score = 20 if "eval(" not in user_code and "exec(" not in user_code else 0
        total_score = (correctness * 50) + readability_score + efficiency_score + security_score

        feedback = []
        if correctness == 0:
            feedback.append("❌ Error in Code Execution! Check syntax or logic errors.")
            feedback.append(f"📌 Error Details: {error_message}")
        else:
            feedback.append("✅ Code executed successfully!")

        if efficiency_score < 30:
            feedback.append("⚡ Performance Issue: Code took longer to execute. Optimize loops or calculations.")
        if readability_score < 20:
            feedback.append("📖 Readability Issue: Code is lengthy. Break into smaller functions.")
        if security_score == 0:
            feedback.append("🔒 Security Risk: Avoid using eval() or exec().")

        return {
            "status": "success" if correctness else "error",
            "execution_time": round(exec_time, 3) if correctness else None,
            "score": max(0, min(100, total_score)),
            "feedback": "\n".join(feedback),
            "error_details": error_message if not correctness else None
        }
    except Exception as e:
        logger.error(f"Error in evaluate_code: {str(e)}")
        return {"status": "error", "message": str(e), "score": 0}

# Optimize code using LLM
def optimize_code_ai(user_code: str, lang: str) -> str:
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                cache_dir=str(CACHE_DIR)
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                token=HF_TOKEN,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                cache_dir=str(CACHE_DIR)
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    try:
        if lang == "python":
            user_code = autopep8.fix_code(user_code)
            user_code = re.sub(r"eval\((.*)\)", r"int(\1)  # Removed eval for security", user_code)
            user_code = re.sub(r"/ 0", "/ 1  # Fixed division by zero", user_code)

        prompt = f"Optimize this {lang} code:\n
{lang}\n{user_code}\n
\nOptimized version:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=1024)

        optimized_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        code_match = re.search(r'
(?:python)?\n(.*?)\n
', optimized_code, re.DOTALL)
        if code_match:
            optimized_code = code_match.group(1)

        return optimized_code if optimized_code else user_code
    except Exception as e:
        logger.error(f"Error in optimize_code_ai: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI optimization failed: {str(e)}")

# Endpoints
@app.post("/evaluate")
async def evaluate_endpoint(request: CodeRequest):
    try:
        result = evaluate_code(request.code, request.language)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error in evaluate_endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize")
async def optimize_endpoint(request: CodeRequest):
    try:
        optimized = optimize_code_ai(request.code, request.language)
        return {"status": "success", "optimized_code": optimized}
    except Exception as e:
        logger.error(f"Error in optimize_endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.options("/evaluate")
async def options_evaluate():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.options("/optimize")
async def options_optimize():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded yet"}
    return {"status": "success", "message": "Model loaded"}

@app.get("/")
async def root():
    return {
        "message": "Code Evaluation API is running",
        "endpoints": {
            "evaluate": "POST /evaluate",
            "optimize": "POST /optimize"
        }
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        timeout_keep_alive=60
    ) 

