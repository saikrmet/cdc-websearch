from pydantic import BaseModel, Field
from typing import Literal, Optional, Any

class AgentRequest(BaseModel):
    thread_id: int
    message: str


class StreamEventBase(BaseModel):
    type: str


class CreateThreadEvent(StreamEventBase):
    type: Literal["CreateThread"]
    thread_id: str


class MessageDeltaEvent(StreamEventBase):
    type: Literal["MessageDelta"]
    message_id: str
    text: str


class ErrorEvent(StreamEventBase):
    type: Literal["ThreadRunError", "RunStepError", "UnhandledEventError", "Error"]
    message: str
    code: Optional[str] = None
    event_type: Optional[str] = None

class RunStepEvent(StreamEventBase):
    type: Literal["RunStep"]
    step_type: Literal["tool_calls", "message_creation"]
    step_details_type: Literal["bing_custom_search"]
    content: str

class ThreadMessageEvent(StreamEventBase):
    type: Literal["ThreadMessage"]
    message_id: str
    role: Literal["user", "assistant"]
    content: Optional[str] = None  # the actual text to show
    status: Optional[str] = None   # "completed", "in_progress", etc.