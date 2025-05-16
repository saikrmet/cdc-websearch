from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from azure.ai.projects.models import MessageRole

class AgentRequest(BaseModel):
    thread_id: Optional[str] = None
    agent_id: str
    message: str

class StreamEventBase(BaseModel):
    type: str

class CreateThreadEvent(StreamEventBase):
    type: Literal["create_thread"]
    thread_id: str

class DeleteThreadRequest(BaseModel):
    thread_id: str

class MessageEvent(StreamEventBase):
    type: Literal["text"]
    role: Literal[MessageRole.AGENT, MessageRole.USER]
    message: str


class ErrorEvent(StreamEventBase):
    type: Literal["error"]
    message: str
    code: str
    event_type: Optional[str] = None

class Citation(StreamEventBase):
    type: Literal["url_citation"]
    title: str
    url: str
    start_index: int
    end_index: int

class BingGroundingEvent(StreamEventBase):
    type: Literal["bing_grounding"]
    title: str
    url: str

class CitationsEvent(StreamEventBase):  
    type: Literal["citations_event"]
    citations: list[Citation]

