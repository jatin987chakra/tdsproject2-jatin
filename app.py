import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import subprocess
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class LLMWithFallback:
    """LLM with fallback from Gemini to OpenAI/OpenRouter."""
    def __init__(self):
        self.gemini_key = None
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        # Try to get first available Gemini key from gemini_api_1 through gemini_api_10
        for i in range(1, 11):
            key_name = f"gemini_api_{i}"
            key = os.getenv(key_name)
            if key and key != "your_api_key_here":
                self.gemini_key = key
                break
        
        # Initialize LLMs - priority order: Gemini, OpenRouter, OpenAI
        if self.gemini_key:
            try:
                self.primary_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.gemini_key, temperature=0.1)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.primary_llm = None
        else:
            self.primary_llm = None
            
        if not self.primary_llm and self.openrouter_key:
            try:
                self.primary_llm = ChatOpenAI(
                    model="gpt-4-turbo",
                    api_key=self.openrouter_key,
                    base_url="https://openrouter.io/api/v1",
                    temperature=0.1
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {e}")
                self.primary_llm = None
                
        if not self.primary_llm and self.openai_key:
            try:
                self.primary_llm = ChatOpenAI(model="gpt-4-turbo", api_key=self.openai_key, temperature=0.1)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
                self.primary_llm = None
        
        if not self.primary_llm:
            raise ValueError("No LLM API key found. Please set GEMINI_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY")
    
    def get_llm(self):
        return self.primary_llm

llm_handler = LLMWithFallback()
llm = llm_handler.get_llm()

@tool
def scrape_url_to_dataframe(url: str) -> str:
    """Scrapes a given URL and returns tabular data as a CSV string."""
    try:
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        tables = soup.find_all('table')
        if not tables:
            text_content = soup.get_text()
            return text_content[:2000]
        
        dataframes = []
        for table in tables:
            try:
                df = pd.read_html(str(table))[0]
                dataframes.append(df)
            except:
                continue
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            return combined_df.to_csv(index=False)
        else:
            text_content = soup.get_text()
            return text_content[:2000]
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

def parse_keys_and_types(variables_str: str) -> dict:
    """Parse variable assignments and determine types."""
    if not variables_str:
        return {}
    
    variables = {}
    for line in variables_str.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            try:
                variables[key] = eval(value)
            except:
                variables[key] = value
    
    return variables

def clean_llm_output(text: str) -> str:
    """Clean LLM output to extract Python code."""
    if '```python' in text:
        code = text.split('```python')[1].split('```')[0]
    elif '```' in text:
        code = text.split('```')[1].split('```')[0]
    else:
        code = text
    
    return code.strip()

def write_and_run_temp_python(code: str, variables: dict = None) -> tuple:
    """Write Python code to temp file and execute."""
    if variables is None:
        variables = {}
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        os.unlink(temp_file)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Execution timeout"
    except Exception as e:
        return 1, "", str(e)

def run_agent_safely_unified(quiz_url: str, email: str, secret: str) -> dict:
    """Run the agent to solve the quiz and return results."""
    try:
        tools = [scrape_url_to_dataframe]
        prompt = hub.pull("hwchase17/react-chat-json")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=15,
            handle_parsing_errors=True
        )
        
        input_text = f"""Visit and analyze the URL: {quiz_url}
        Extract ALL data from the URL. Return results as JSON with:
        {{
            "email": "{email}",
            "secret": "{secret}",
            "analysis": <detailed analysis>,
            "answers": <quiz answers if present>,
            "data": <extracted data>
        }}"""
        
        result = agent_executor.invoke({"input": input_text})
        return {"status": "success", "output": result.get("output", "")}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"status": "error", "message": str(e)}

class QuizRequest(BaseModel):
    email: str
    secret: str
    quiz_url: str

@app.get("/")
async def root():
    """Root endpoint showing API documentation."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TDS Project 2 - LLM Quiz Solver</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f4f4f4; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }
            code { background: #ddd; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>TDS Project 2 - LLM Quiz Solver</h1>
        <h2>API Endpoints</h2>
        <div class="endpoint">
            <h3>POST /api</h3>
            <p>Solve quiz from URL using LLM</p>
            <p>Request body:</p>
            <code>{
                "email": "user@example.com",
                "secret": "your_secret",
                "quiz_url": "https://example.com/quiz"
            }</code>
            <p>Responses: 200 (success), 400 (bad request), 403 (invalid secret)</p>
        </div>
        <div class="endpoint">
            <h3>GET /api</h3>
            <p>Returns: {"status": "ok"}</p>
        </div>
        <p>GitHub: <a href="https://github.com/jatin987chakra/tdsproject2-jatin">jatin987chakra/tdsproject2-jatin</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api")
async def solve_quiz(request: QuizRequest):
    """POST endpoint to solve quiz."""
    try:
        if not request.email or not request.quiz_url or not request.secret:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Verify secret
        expected_secret = os.getenv("SECRET_KEY")
        if request.secret != expected_secret:
            raise HTTPException(status_code=403, detail="Invalid secret")
        
        # Run agent
        result = run_agent_safely_unified(request.quiz_url, request.email, request.secret)
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /api: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api")
async def get_api():
    """GET endpoint for API status."""
    return {"status": "ok"}

@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
