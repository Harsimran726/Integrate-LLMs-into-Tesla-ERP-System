# main.py
from fastapi.responses import HTMLResponse, JSONResponse  # Change to JSONResponse
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from rag_erp import user_input  # Import your RAG processing function
import markdown  
from fastapi.staticfiles import StaticFiles
import os
# Initialize FastAPI app
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
# Set up templates directory for Jinja2
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Route for landing page
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("landing-page.html", {"request": request})

# Route for the chat page
@app.get("/chats", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Route for handling chat interactions
@app.post("/chat", response_class=HTMLResponse)
async def chat_response(request: Request, text: str = Form(...)):
    # Call the RAG processing function from rag_erp.py
    output = user_input(text)
    output_html = markdown.markdown(output)
    print("Here is the ourtput")
    return JSONResponse(content={"output": output_html})

# To run the app, use the terminal:
# uvicorn main:app --reload
