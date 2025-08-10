import os
import sys
import io
import json
import requests
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import contextlib

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

So make sure not to use any library which would need installation

---DATA---
{scraped_data}

---QUESTIONS---
{questions}

RULES:
1. The output must be a JSON object
2. Your job is to provide the code that will answer these questions.
3. Answer should be in the structure of response of given in question.
4. If csv file sent with POST required, it is present in "sent_csv/data.csv"
5. If image sent with POST required, it is present in "sent_image/image.png"
6. Construct the final JSON object using JSON.dumps()
7. Assign the JSON string to a variable named output (do not print)
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
@app.post("/api/")
async def upload_file(
    questions: UploadFile = File(..., alias="questions.txt"),
    image: Optional[UploadFile] = File(None, alias="image.png"),
    data: Optional[UploadFile] = File(None, alias="data.csv")
):
    try:
        text = (await questions.read()).decode("utf-8")

        if image:
            img_bytes = await image.read()
            Image.open(io.BytesIO(img_bytes))  # just to validate

        if data:
            csv_bytes = await data.read()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            scraped_data = df.to_csv(index=False)
            task_file = answer_questions_with_gemini(text, scraped_data)
            return run_with_retry(task_file)

        scraper_path = generate_scraping_code(text)
        run_with_retry(scraper_path)
        scraped_file = os.path.join(TMP_DIR, "scraped_data.txt")
        with open(scraped_file, "r") as f:
            scraped_data = f.read()
        task_file = answer_questions_with_gemini(text, scraped_data)
        return run_with_retry(task_file)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Hello!"}
