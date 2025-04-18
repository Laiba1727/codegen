from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import autopep8
import subprocess
import time
import re
import os
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Code Evaluation & Optimization API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment Setup
CACHE_DIR = Path("/.cache/huggingface")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)

# Lazy model loading with efficient memory management
MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"

tokenizer = None
model = None

def load_model():
    """Load the model only when it's needed to save memory."""
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir=str(CACHE_DIR))
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16,  # Use float16 for memory optimization
                cache_dir=str(CACHE_DIR))
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

# Request Model
class CodeRequest(BaseModel):
    code: str
    language: str = "python"

# Helper Functions
def evaluate_code(user_code: str, lang: str) -> dict:
    """Evaluate code for correctness, performance, and security"""
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

    try:
        if lang in commands:
            result = subprocess.run(" ".join(commands[lang]), 
                                capture_output=True, 
                                text=True, 
                                timeout=5, 
                                shell=True)
            exec_time = time.time() - start_time
            correctness = 1 if result.returncode == 0 else 0
            error_message = None if correctness else result.stderr.strip()
        else:
            return {"status": "error", "message": "Unsupported language", "score": 0}
    except Exception as e:
        return {"status": "error", "message": str(e), "score": 0}

    # Scoring logic
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

def optimize_code_ai(user_code: str, lang: str) -> str:
    """Generate optimized code using AI"""
    load_model()  # Load the model only when optimization is requested
    try:
        if lang == "python":
            user_code = autopep8.fix_code(user_code)
            user_code = re.sub(r"eval\((.*)\)", r"int(\1)  # Removed eval for security", user_code)
            user_code = re.sub(r"/ 0", "/ 1  # Fixed division by zero", user_code)
        
        prompt = f"Optimize this {lang} code:\n```{lang}\n{user_code}\n```\nOptimized version:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():  # Avoid unnecessary memory use
            outputs = model.generate(**inputs, max_length=1024)
        
        optimized_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        code_match = re.search(r'```(?:python)?\n(.*?)\n```', optimized_code, re.DOTALL)
        if code_match:
            optimized_code = code_match.group(1)
        
        return optimized_code if optimized_code else user_code
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI optimization failed: {str(e)}")

# API Endpoints
@app.post("/evaluate")
async def evaluate_endpoint(request: CodeRequest):
    try:
        result = evaluate_code(request.code, request.language)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize")
async def optimize_endpoint(request: CodeRequest):
    try:
        optimized = optimize_code_ai(request.code, request.language)
        return {"status": "success", "optimized_code": optimized}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "API is running",
        "model": MODEL_NAME,
        "endpoints": {
            "evaluate": "POST /evaluate",
            "optimize": "POST /optimize"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))  # Railway will provide PORT
    uvicorn.run("app:app", host="0.0.0.0", port=port)


