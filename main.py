import os
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import ValidationError
from contextlib import asynccontextmanager
from pathlib import Path
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import Agent, BingCustomSearchTool, AsyncFunctionTool, AsyncToolSet
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from jinja2 import Jinja2Templates

from models import AgentRequest
from agent import stream_agent_response, format_as_ndjson


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    async with DefaultAzureCredential() as creds:
        async with AIProjectClient.from_connection_string(
            credential=creds,
            conn_str=os.getenv("PROJECT_CONNECTION_STRING"),
        ) as project_client:
            app.state.agent = await project_client.get_agent(agent_id=os.getenv("AGENT_ID"))
            yield

    
app = FastAPI(
    title="CDC Custom Web Search", 
    description="An agentic AI chatbot that searches the web from CDC approved domains",
    lifespan=lifespan
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_agent(request: Request) -> Agent:
    return request.app.state.agent

@app.get("/", response_class=HTMLResponse)
async def load_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/", response_class=StreamingResponse)
async def get_chat_response(request: Request, agent = Depends(get_agent)):
    try:
        data = await request.json()
        agent_request = AgentRequest(**data)
    except ValidationError as e:
        return JSONResponse(
            status_code=422,
            content={"type": "request_validation", "detail": e.errors()}
        )

    return StreamingResponse(
        ## CHECK PARAMS
        format_as_ndjson(stream_agent_response(agent, agent_request)),
        media_type="application/x-ndjson"
    )       


