import os
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from pydantic import ValidationError
from contextlib import asynccontextmanager
from pathlib import Path
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import Agent, BingCustomSearchTool, AsyncFunctionTool, AsyncToolSet
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
import logging
from models import AgentRequest, DeleteThreadRequest
from agent import stream_agent_response, format_as_ndjson, delete_thread


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"[{os.getenv("PROJECT_CONNECTION_STRING")}]")
    async with DefaultAzureCredential() as creds:
        ## Authenticate and get from KV in app
        async with AIProjectClient.from_connection_string(
            credential=creds,
            conn_str=os.getenv("PROJECT_CONNECTION_STRING"),
        ) as project_client:
            app.state.project_client = project_client
            yield

    
app = FastAPI(
    title="CDC Web Search", 
    description="An agentic AI chatbot that searches the web from CDC approved domains",
    lifespan=lifespan
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_project_client(request: Request) -> AIProjectClient:
    return request.app.state.project_client

@app.get("/")
async def home():
    logger.info("Redirect to chat")
    return RedirectResponse(url="/chat")

@app.get("/agents")
async def get_agents(project_client: AIProjectClient = Depends(get_project_client)):
    agents = await project_client.agents.list_agents()
    agent_map = [{"id": agent.id, "name": agent.name} for agent in agents.data]
    logger.info(f"Agents: {agent_map}")
    return agent_map

@app.get("/chat", response_class=HTMLResponse)
async def load_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat", response_class=StreamingResponse)
async def get_chat_response(request: AgentRequest, project_client = Depends(get_project_client)):
    return StreamingResponse(
        format_as_ndjson(stream_agent_response(request, project_client)),
        media_type="application/x-ndjson"
    )

@app.post("/delete_thread", response_class=JSONResponse)
async def delete_thread_request(request: DeleteThreadRequest, project_client: AIProjectClient = Depends(get_project_client)):
    thread_id = request.thread_id
    try:
        await project_client.agents.delete_thread(thread_id=thread_id)
        logger.info(f"Thread {thread_id} deleted successfully.")
        return {"message": f"Thread {thread_id} deleted successfully."}
    except Exception as e:
        logger.error("Thread deletion failed for %s: %s", thread_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to delete thread {thread_id}: {str(e)}")
