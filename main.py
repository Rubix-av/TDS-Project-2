from fastapi import FastAPI,File, UploadFile    
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from dotenv import load_dotenv
from typing import Optional
from PIL import Image
import pandas as pd
import subprocess
import requests
import json
import sys
import io
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"]) # Allow GET requests from all origins
# Or, provide more granular control:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow a specific domain
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow specific methods
    allow_headers=["*"],  # Allow all headers
)

def run_scraper():
    result = subprocess.run(
        [sys.executable, "scraper.txt"],
        capture_output=True,
        text=True
    )
    return result

def fix_code_with_llm(code: str, error: str) -> str:
    """
    Sends the failing code and error message to the LLM for fixing.
    Returns the fixed code.
    """
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

def run_scraper_with_retry(max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        print(f"Scraper {attempt}...")
        result = run_scraper()

        if result.returncode == 0:
            print("Code ran successfully.")
            return result.stdout
        
        print(f"Error:\n{result.stderr}")
        with open("scraper.txt", "r") as f:
            current_code = f.read()

        fixed_code = fix_code_with_llm(current_code, result.stderr)
        with open("scraper.txt", "w") as f:
            f.write(fixed_code)

        print("🔄 Code updated from LLM, retrying...")

    print("❌ All attempts failed.")
    return None

def generate_scraping_code(task: str):

    # Load the base scraping prompt
    task_breakdown_file = os.path.join('prompts', "scrape_data.txt")
    with open(task_breakdown_file, 'r') as f:
        task_breakdown_prompt = f.read()

    try:
        
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

        data = json.loads(response.text)
        content = data["choices"][0]["message"]["content"]
        
        cleaned_content = content.replace("```python", "").replace("```", "").strip()

        # Save the generated code
        with open("scraper.txt", "w") as f:
            f.write(cleaned_content)

        return cleaned_content

    except Exception as e:
        print(f"Gemini request failed: {e}")
        return None


def run_coded_task():
    """Runs coded_task.txt with the system Python and returns the subprocess result."""
    result = subprocess.run(
        [sys.executable, "code/coded_task.txt"],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print("Output:\n", result.stdout)
    if result.stderr:
        print("Errors:\n", result.stderr)

    return result


def fix_coded_task_with_gemini(code: str, error: str) -> str:
    """
    Sends failing code + error to Gemini for fixing.
    Returns the fixed code.
    """
    prompt = f"""
You are a Python expert. Fix the given code based on the error.
Return only the corrected Python code without explanations or markdown formatting.
Construct the final JSON object using JSON.dumps() (as a Python dictionary)

---CODE---
{code}

---ERROR---
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
        print(f"Gemini request failed: {e}")
        return code


def run_coded_task_with_retry(max_attempts=5):
    """
    Runs coded_task.txt up to `max_attempts` times.
    If it fails, sends to Gemini for fixing and retries.
    """
    for attempt in range(1, max_attempts + 1):
        print(f"Coded task {attempt}...")
        result = run_coded_task()

        if result.returncode == 0:
            print("✅ Code ran successfully.")
            return result.stdout

        print(f"❌ Error:\n{result.stderr}")

        # Read current code
        with open("code/coded_task.txt", "r") as f:
            current_code = f.read()

        # Fix with Gemini
        fixed_code = fix_coded_task_with_gemini(current_code, result.stderr)

        # Save updated code
        with open("code/coded_task.txt", "w") as f:
            f.write(fixed_code)

        print("🔄 Code updated from Gemini, retrying...")

    print("❌ All attempts failed.")
    return None

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
3. Answer should be in the structure of response of given in question. Keep only the answers, not any extra text. 4. It should only be in the case when explicitly mentioned.
5. If csv file sent with POST required, it is present in "sent_csv/data.csv"
6. If image sent with POST required, it is present in "sent_image/image.png"
7. Construct the final JSON object using JSON.dumps()
"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )
    
    content = response.text
    cleaned_content = content.replace("```python", "").replace("```", "").strip()
    
    with open("code/coded_task.txt", "w") as f:
        f.write(cleaned_content)
    
    return cleaned_content
    
@app.post("/api/")
async def upload_file(
    questions: UploadFile = File(..., alias="questions.txt"),
    image: Optional[UploadFile] = File(None, alias="image.png"),
    data: Optional[UploadFile] = File(None, alias="data.csv")
):
    try:
        # Read questions.txt
        content = await questions.read()
        text = content.decode("utf-8")
        print("Questions content:", text)

        # Read image.png if provided
        img_obj = None
        if image:
            img_bytes = await image.read()
            img_obj = Image.open(io.BytesIO(img_bytes))
            print(f"Image loaded: {img_obj.format}, size={img_obj.size}")

        # If CSV is provided, skip scraping
        if data:
            csv_bytes = await data.read()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # Pass CSV content as 'scraped_data'
            scraped_data = df.to_csv(index=False)
            answer_questions_with_gemini(text, scraped_data)
            final_answer = run_coded_task_with_retry()
            return final_answer

        # Otherwise, do scraping workflow
        generate_scraping_code(text)
        
        # Run the scraper with auto-retries
        run_scraper_with_retry(max_attempts=3)
        
        with open("scraped_data.txt", 'r') as f:
            scraped_data = f.read()
        
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        print("----------------------------------------------")
        print(scraped_data)
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        print("-----------------------------------------------")

        # Generate final answer
        answer_questions_with_gemini(text, scraped_data)
        final_answer = run_coded_task_with_retry()
        
        return final_answer

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "Hello!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
        
    