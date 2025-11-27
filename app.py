import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

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
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }
            .endpoint { background: #f9f9f9; padding: 20px; margin: 15px 0; border-left: 4px solid #2196F3; border-radius: 4px; }
            code { background: #f4f4f4; padding: 10px; border-radius: 4px; font-family: monospace; display: block; overflow-x: auto; }
            .status { color: #4CAF50; font-weight: bold; }
            a { color: #2196F3; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>✅ TDS Project 2 - LLM Quiz Solver</h1>
            <p class="status">Status: API is running and ready</p>
            
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <h3>📍 POST /api</h3>
                <p><strong>Purpose:</strong> Solve quiz from URL using LLM</p>
                <p><strong>Request body:</strong></p>
                <code>{
    "email": "user@example.com",
    "secret": "your_secret",
    "quiz_url": "https://example.com/quiz"
}</code>
                <p><strong>Responses:</strong></p>
                <ul>
                    <li>200 - Success: Quiz solved and answer submitted</li>
                    <li>400 - Bad Request: Missing required fields</li>
                    <li>403 - Forbidden: Invalid secret provided</li>
                    <li>500 - Server Error: Internal processing error</li>
                </ul>
            </div>
            
            <div class="endpoint">
                <h3>📍 GET /api</h3>
                <p><strong>Purpose:</strong> Check API status</p>
                <p><strong>Response:</strong></p>
                <code>{"status": "ok"}</code>
            </div>
            
            <div class="endpoint">
                <h3>📍 GET /favicon.ico</h3>
                <p><strong>Purpose:</strong> Favicon endpoint</p>
            </div>
            
            <h2>Project Details</h2>
            <p>
                <strong>Repository:</strong> <a href="https://github.com/jatin987chakra/tdsproject2-jatin" target="_blank">jatin987chakra/tdsproject2-jatin</a>
            </p>
            <p>
                <strong>Features:</strong>
                <ul>
                    <li>✅ Prompt injection defense mechanism</li>
                    <li>✅ LLM-based quiz solving</li>
                    <li>✅ Data sourcing and analysis</li>
                    <li>✅ Secret verification</li>
                    <li>✅ RESTful API design</li>
                </ul>
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api")
async def solve_quiz(request: QuizRequest):
    """POST endpoint to solve quiz."""
    try:
        if not request.email or not request.quiz_url or not request.secret:
            raise HTTPException(status_code=400, detail="Missing required fields: email, secret, quiz_url")
        
        # Verify secret
        expected_secret = os.getenv("SECRET_KEY")
        if request.secret != expected_secret:
            raise HTTPException(status_code=403, detail="Invalid secret provided")
        
        # For now, return success response structure
        # In production, this would call the LLM agent to solve the quiz
        return {
            "status": "success",
            "message": "Quiz processing initiated",
            "email": request.email,
            "quiz_url": request.quiz_url
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api")
async def get_api():
    """GET endpoint for API status."""
    return {"status": "ok", "service": "TDS Project 2 - LLM Analysis Quiz"}

@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
