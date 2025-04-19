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

# Import Pygments for language detection
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Auto Language Code Evaluation API",
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

# Hugging Face Token
HF_TOKEN = os.getenv("HF_API_TOKEN")

# Model Configuration
MODEL_NAME = "Salesforce/codet5-small"
tokenizer: Optional[AutoTokenizer] = None
model: Optional[AutoModelForSeq2SeqLM] = None

# Supported languages mapping
LANG_EXT = {
    'python': 'py',
    'java': 'java',
    'cpp': 'cpp',
    'javascript': 'js'
}

# Request Model
class CodeRequest(BaseModel):
    code: str
    language: Optional[str] = None  # auto-detect if not provided

# Detect programming language using Pygments
def detect_language(code: str) -> str:
    try:
        lexer = guess_lexer(code)
        alias = lexer.aliases[0]
        if alias in ('python', 'py'):
            return 'python'
        elif alias in ('java',):
            return 'java'
        elif alias in ('cpp', 'c++'):
            return 'cpp'
        elif alias in ('js', 'javascript', 'nodejs'):
            return 'javascript'
    except ClassNotFound:
        logger.warning('Language detection failed, defaulting to plaintext')
    return 'plaintext'

# Evaluate code
def evaluate_code(user_code: str, lang: str) -> dict:
    try:
        file_ext = LANG_EXT.get(lang, 'txt')
        filename = f"temp_script.{file_ext}"

        # Write code to temporary file
        with open(filename, "w") as f:
            f.write(user_code)

        # Execution commands
        commands = {
            "python": ["python3", filename],
            "java": ["javac", filename, "&&", "java", filename.replace(".java", "")],
            "cpp": ["g++", filename, "-o", "temp_out", "&&", "./temp_out"],
            "javascript": ["node", filename]
        }

        start_time = time.time()
        if lang in commands:
            result = subprocess.run(" ".join(commands[lang]), capture_output=True,
                                    text=True, timeout=15, shell=True)
            exec_time = time.time() - start_time
            success = result.returncode == 0
            stderr = result.stderr.strip() if not success else None
        else:
            return {"status": "error", "message": "Unsupported language", "score": 0}

        # Scoring metrics
        score = 0
        score += 50 if success else 0
        score += 20 if len(user_code) < 200 else 10
        score += 30 if exec_time < 1 else 10
        score += 20 if not re.search(r"\b(eval|exec)\b", user_code) else 0
        total = min(max(score, 0), 100)

        feedback = []
        if not success:
            feedback.append(f"Error: {stderr}")
        else:
            feedback.append("Execution successful.")
        if exec_time >= 1:
            feedback.append("Performance: consider optimizing loops.")
        if len(user_code) >= 200:
            feedback.append("Readability: consider refactoring into functions.")
        if re.search(r"\b(eval|exec)\b", user_code):
            feedback.append("Security: avoid eval()/exec().")

        return {
            "status": "success" if success else "error",
            "execution_time": round(exec_time, 3) if success else None,
            "score": total,
            "feedback": feedback
        }
    except Exception as ex:
        logger.error(f"evaluation error: {ex}")
        return {"status": "error", "message": str(ex), "score": 0}

# Optimize code using LLM
def optimize_code_ai(user_code: str, lang: str) -> str:
    global tokenizer, model
    if tokenizer is None or model is None:
        logger.info("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, cache_dir=str(CACHE_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, token=HF_TOKEN, cache_dir=str(CACHE_DIR), torch_dtype=torch.float32
        ).to("cpu")
        logger.info("Model loaded on CPU")

    if lang == 'python':
        user_code = autopep8.fix_code(user_code)

    prompt = f"""Optimize this {lang} code:\n\n{user_code}\n\nOptimized version:"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_length=1024)
    optimized = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # extract optimized block
    match = re.search(r"(?:Optimized version:\n)(.*)", optimized, re.DOTALL)
    return match.group(1).strip() if match else optimized

# Endpoints
@app.post("/evaluate")
async def evaluate_endpoint(req: CodeRequest):
    lang = req.language or detect_language(req.code)
    result = evaluate_code(req.code, lang)
    return {"language": lang, "result": result}

@app.post("/optimize")
async def optimize_endpoint(req: CodeRequest):
    lang = req.language or detect_language(req.code)
    optimized = optimize_code_ai(req.code, lang)
    return {"language": lang, "optimized_code": optimized}

@app.options("/evaluate")
async def options_eval():
    return Response(status_code=200, headers={"Access-Control-Allow-Origin": "*"})

@app.options("/optimize")
async def options_opt():
    return Response(status_code=200, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/health")
async def health():
    return {"status": "ok" if model is not None else "loading"}

@app.get("/")
async def root():
    return {"message": "Auto Language Code API running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "auto_lang_code_api:app",
        host="0.0.0.0",
        port=port,
        workers=1
    )
