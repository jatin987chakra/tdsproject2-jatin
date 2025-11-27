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

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from langchain_core.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global LLM instance (lazy-loaded)
_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        _llm_instance = ChatOpenAI(
            model="gpt-4-turbo",
            api_key=api_key,
            base_url="https://openrouter.io/api/v1",
            temperature=0.1
        )
    return _llm_instance

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

def run_agent_safely_unified(quiz_url: str, email: str, secret: str) -> dict:
    """Run the agent to solve the quiz and return results."""
    try:
        llm = get_llm()
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
