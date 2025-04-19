from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import autopep8
import subprocess
import time
import re
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional
import requests

# Optional Pygments import for language detection
try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
    _pygments_available = True
except ImportError:
    _pygments_available = False

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

# Detect programming language
def detect_language(code: str) -> str:
    if not _pygments_available:
        logger.warning('Pygments not installed; defaulting to python')
        return 'python'
    try:
        lexer = guess_lexer(code)
        alias = lexer.aliases[0]
        if alias in ('python', 'py'):
            return 'python'
        elif alias == 'java':
            return 'java'
        elif alias in ('cpp', 'c++'):
            return 'cpp'
        elif alias in ('js', 'javascript', 'nodejs'):
            return 'javascript'
    except Exception as ex:
        logger.warning(f'Language detection failed ({ex}); defaulting to python')
    return 'python'

# Evaluate code
def evaluate_code(user_code: str, lang: str) -> dict:
    file_ext = LANG_EXT.get(lang, 'txt')
    filename = f"temp_script.{file_ext}"
    try:
        with open(filename, "w") as f:
            f.write(user_code)

        commands = {
            'python': ['python3', filename],
            'java': ['javac', filename, '&&', 'java', filename.replace('.java','')],
            'cpp': ['g++', filename, '-o', 'temp_out', '&&', './temp_out'],
            'javascript': ['node', filename]
        }

        start_time = time.time()
        if lang in commands:
            proc = subprocess.run(' '.join(commands[lang]), capture_output=True,
                                   text=True, timeout=15, shell=True)
            exec_time = time.time() - start_time
            success = proc.returncode == 0
            stderr = proc.stderr.strip() if not success else None
        else:
            return {'status':'error','message':'Unsupported language','score':0}

        score = 0
        score += 50 if success else 0
        score += 20 if len(user_code) < 200 else 10
        score += 30 if exec_time < 1 else 10
        score += 20 if not re.search(r"\b(eval|exec)\b", user_code) else 0
        total = max(0, min(score,100))

        feedback = []
        if not success:
            feedback.append(f"Error: {stderr}")
        else:
            feedback.append('Execution successful.')
        if exec_time >= 1:
            feedback.append('Performance: consider optimizing loops.')
        if len(user_code) >= 200:
            feedback.append('Readability: refactor into functions.')
        if re.search(r"\b(eval|exec)\b", user_code):
            feedback.append('Security: avoid eval()/exec().')

        return {
            'status':'success' if success else 'error',
            'execution_time':round(exec_time,3) if success else None,
            'score':total,
            'feedback':feedback
        }
    except Exception as ex:
        logger.error(f"evaluation error: {ex}")
        return {'status':'error','message':str(ex),'score':0}

def fallback_optimize_code(code: str, lang: str) -> str:
    if lang == 'python':
        optimized = autopep8.fix_code(code)
        optimized += "\n# TIP: Break down large functions for readability."
        return optimized
    elif lang == 'java':
        optimized = re.sub(r'{', '{\n', code)
        optimized = re.sub(r';', ';\n', optimized)
        optimized += "\n// TIP: Use streams and avoid nested loops when possible."
        return optimized
    elif lang == 'cpp':
        optimized = re.sub(r'{', '{\n', code)
        optimized = re.sub(r';', ';\n', optimized)
        optimized += "\n// TIP: Consider STL algorithms and minimize pointer usage."
        return optimized
    return code + "\n# No optimization available for this language."

# Optimize code using Hugging Face API
def optimize_code_ai(user_code: str, lang: str) -> str:
    try:
        prompt = f"Improve and optimize this {lang} code:\n\n{user_code}\n\nOptimized version:\n"
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256}
        }
        response = requests.post(
            f"https://api-inference.huggingface.co/models/Salesforce/codet5-small",
            headers=headers,
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            generated = result[0]["generated_text"]
            if 'Optimized version:' in generated:
                return generated.split('Optimized version:')[-1].strip()
            return generated.strip()
        else:
            raise ValueError("Unexpected response format")

    except Exception as ex:
        logger.warning(f"Hugging Face API failed: {ex}")
        return fallback_optimize_code(user_code, lang)

# Endpoints
@app.post('/evaluate')
async def evaluate_endpoint(req: CodeRequest):
    lang = req.language or detect_language(req.code)
    logger.info(f'Evaluate request language: {lang}')
    result = evaluate_code(req.code, lang)
    return {'language':lang, 'result':result}

@app.post('/optimize')
async def optimize_endpoint(req: CodeRequest):
    lang = req.language or detect_language(req.code)
    logger.info(f'Optimize endpoint called; language: {lang}')
    optimized_code = optimize_code_ai(req.code, lang)
    return {'language':lang, 'optimized_code':optimized_code}

@app.options('/evaluate')
async def options_eval():
    return Response(status_code=200, headers={'Access-Control-Allow-Origin':'*'})

@app.options('/optimize')
async def options_opt():
    return Response(status_code=200, headers={'Access-Control-Allow-Origin':'*'})

@app.get('/health')
async def health():
    return {'status':'ok'}

@app.get('/')
async def root():
    return {'message':'Auto Language Code API running'}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT',8080))
    uvicorn.run(
        'auto_lang_code_api:app', host='0.0.0.0', port=port, workers=1
    )
