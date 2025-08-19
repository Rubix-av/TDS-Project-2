import os
import re
import io
import json
import requests
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = "/tmp"
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# ------------------
# Utility: run code in same process (capture stdout + output var)
# ------------------
def run_python_code_in_process(file_path: str):
    """Executes a Python script file in the current process."""
    with open(file_path, "r") as f:
        code = f.read()
    exec_globals = {}
    try:
        exec(code, exec_globals)
        output = exec_globals.get("output", None)

        # If output is a JSON string, parse into Python object
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass

        return output
    except Exception as e:
        raise RuntimeError(str(e))

# ------------------
# Code fixing with Gemini
# ------------------
def fix_code_with_llm(code: str, error: str) -> str:
    prompt = f"""You are a Python expert.
Fix the given code based on the error.
Return only the corrected Python code without explanations or markdown formatting.

Code:
{code}

Error:
{error}
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )
        fixed_code = response.text
        return fixed_code.replace("```python", "").replace("```", "").strip()
    except Exception as e:
        print(f"LLM request failed: {e}")
        return code

# ------------------
# Retry runner
# ------------------
def run_with_retry(file_path: str, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        print(f"Run attempt {attempt}...")
        try:
            return run_python_code_in_process(file_path)
        except Exception as e:
            print(f"Error: {e}")
            with open(file_path, "r") as f:
                current_code = f.read()
            fixed_code = fix_code_with_llm(current_code, str(e))
            with open(file_path, "w") as f:
                f.write(fixed_code)
            print("🔄 Code updated, retrying...")
    raise RuntimeError("All attempts failed.")

# ------------------
# Scraper code generation
# ------------------
def generate_scraping_code(task: str):
    task_breakdown_file = os.path.join(PROMPTS_DIR, "scrape_data.txt")
    with open(task_breakdown_file, "r") as f:
        task_breakdown_prompt = f.read()
    messages = [
        {"role": "system", "content": "You are a professional web scraper. Do not solve or answer questions."},
        {"role": "user", "content": f"{task}\n\n{task_breakdown_prompt}"}
    ]
    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "temperature": 0.2
        },
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
    )
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    cleaned_content = content.replace("```python", "").replace("```", "").strip()
    scraper_path = os.path.join(TMP_DIR, "scraper.py")
    with open(scraper_path, "w") as f:
        f.write(cleaned_content)
    return scraper_path

# ------------------
# Answer questions
# ------------------
def answer_questions_with_gemini(questions: str, scraped_data: str) -> str:
    
    if scraped_data == "No data":
        response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[f"""
                  No data is available to answer the questions, so only return the json object in the same structure as specified in the question below. Also do not add any explanations or additional text. Return the json object as plain text and don't include any code blocks (```python/json```, etc).
                  
                  {questions}
                  """],
        )
        content = response.text
        content = re.sub(r"\s+", " ", content)
        return content

    prompt = f"""
You are a data analyst. Below is the data that has been scraped from the web, and a set of questions that must be answered based on this data. The output must be a JSON object

These are the installed libraries:
fastapi
uvicorn
google-genai
python-multipart
python-dotenv
requests
beautifulsoup4
matplotlib
seaborn
pandas
numpy
scipy

So make sure not to use any library which would need installation.
Don't use networkx library

---DATA---
{scraped_data}

---QUESTIONS---
{questions}

RULES:
1. The output must be a JSON object
2. Your job is to provide the code that will answer these questions.
3. Answer should be in the structure of response of given in question.
4. Construct the final JSON object using JSON.dumps()
5. Assign the JSON string to a variable named output (do not print)
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )
    content = response.text
    cleaned_content = content.replace("```python", "").replace("```", "").strip()
    task_path = os.path.join(TMP_DIR, "coded_task.py")
    with open(task_path, "w") as f:
        f.write(cleaned_content)
    return task_path

# ------------------
# FastAPI endpoints
# ------------------
from fastapi import Request

from typing import List, Tuple
from fastapi import Request, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from PIL import Image
import io, os

@app.post("/api/")
async def upload_file(request: Request, questions: UploadFile = File(None, alias="questions.txt")):
    try:
        form = await request.form()
        files: List[Tuple[str, UploadFile]] = []

        # Collect all file-like objects flexibly
        for key, value in form.multi_items():
            try:
                filename = getattr(value, "filename", None)
                read_method = getattr(value, "read", None)
                if filename and callable(read_method):
                    files.append((key, value))
            except Exception:
                continue

        # Handle questions.txt (or fallback to text field)
        text = None
        if questions:
            text = (await questions.read()).decode("utf-8").strip()
        elif "questions.txt" in form or "question" in form:
            text = str(form.get("questions.txt") or form.get("question") or "").strip()

        if not text:
            return JSONResponse(status_code=400, content={"error": "Missing questions.txt or question text"})

        scraped_data = None

        # Process all other uploaded files dynamically
        for key, file in files:
            if key == "questions.txt":
                continue  # already handled

            filename = file.filename.lower()

            # CSV file
            if filename.endswith(".csv"):
                csv_bytes = await file.read()
                df = pd.read_csv(io.BytesIO(csv_bytes))
                scraped_data = df.to_csv(index=False)

            # Image file
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                img_bytes = await file.read()
                Image.open(io.BytesIO(img_bytes))  # validate

        # If we have CSV data, answer directly
        if scraped_data:
            task_file = answer_questions_with_gemini(text, scraped_data)
            return run_with_retry(task_file, max_attempts=6)

        # Otherwise, try scraping fallback
        try:
            scraper_path = generate_scraping_code(text)
            run_with_retry(scraper_path)
            scraped_file = os.path.join(TMP_DIR, "scraped_data.txt")
            with open(scraped_file, "r") as f:
                scraped_data = f.read()
            task_file = answer_questions_with_gemini(text, scraped_data)
            return run_with_retry(task_file, max_attempts=4)
        except Exception:
            # If scraping fails too
            task_file = answer_questions_with_gemini(text, scraped_data="No data")
            return task_file

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Hello!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5045)
