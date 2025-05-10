from pydantic import BaseModel, Field
from typing import Literal, Optional, Any

class AgentRequest(BaseModel):
    thread_id: int
    message: str


class StreamEventBase(BaseModel):
    type: str


class ThreadInitEvent(StreamEventBase):
    type: Literal["thread_init"]
    thread_id: str


class MessageStreamEvent(StreamEventBase):
    type: Literal["message"]
    role: Literal["user", "assistant", "system"]
    message: str

class ErrorStreamEvent(StreamEventBase):
    type: Literal["error"]
    message: str
    details: Optional[Any] = None
    code: Optional[str] = None  