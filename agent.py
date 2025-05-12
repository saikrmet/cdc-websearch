import asyncio
import logging, json
import re
from typing import AsyncGenerator, Any

from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import Agent, AgentThread, MessageRole
from azure.ai.projects.models import AsyncAgentEventHandler, MessageDeltaChunk, ThreadMessage, ThreadRun, RunStep

from pydantic import ValidationError
from models import AgentRequest, CreateThreadEvent, MessageDeltaEvent, ErrorEvent, ThreadMessageEvent, RunStepEvent

logger = logging.getLogger(__name__)


async def stream_agent_response(agent_request: AgentRequest, project_client: AIProjectClient, agent: Agent):
    thread: AgentThread = None

    if agent_request.thread_id == -1:
        thread = await project_client.agents.create_thread()
        logger.info(f"Created new thread ID: {thread.id}")
    else:
        thread = await project_client.agents.get_thread(agent_request.thread_id)
        logger.info(f"Retrieved thread ID: {thread.id}")
    
    thread_id = thread.id

    create_thread = CreateThreadEvent(type="CreateThread", thread_id=thread_id)
    yield create_thread

    message = await project_client.agents.create_message(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=agent_request.message,
    )

    queue = asyncio.Queue()
    event_handler = StreamingEventHandler(queue)

    async with await project_client.agents.create_stream(
        thread_id=thread_id, 
        agent_id=agent.id,
        event_handler=event_handler
    ) as stream:
        await stream.until_done()
    
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event

    

class StreamingEventHandler(AsyncAgentEventHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue 

    async def on_thread_run(self, run: "ThreadRun") -> None:
        logger.info(f"ThreadRun status: {run.status}")
        if run.status == "failed":
            event = ErrorEvent(
                type="ThreadRunError",
                message=run.last_error.message,
                code=run.last_error.code
            )
            logger.info(f"A ThreadRunError occurred. Event: {event}")
            await self.queue.put(event)

    async def on_error(self, data: str) -> None:
        event = ErrorEvent(
            type="Error",
            message=data
        )
        logger.info(f"An Error occurred. Event: {event}")
        await self.queue.put(event)

    async def on_unhandled_event(self, event_type: str, event_data: Any) -> None:
        event = ErrorEvent(
            type="UnhandledEventError",
            message=event_data,
            event_type=event_type
        )
        logger.info(f"An UnhandledEventError occurred. Event: {event}")
        await self.queue.put(event)

    async def on_done(self) -> None:
        logger.info("Streaming done.")
        await self.queue.put(None)

    async def on_message_delta(self, delta: "MessageDeltaChunk") -> None:
        # Need to get fields and structure Pydantic object
        event = MessageDeltaEvent(
            type="MessageDelta",
            message_id=delta.id,
            text=delta.text
        )
        logger.info(f"MessageDeltaEvent: {event.text}")
        await self.queue.put(event)

    async def on_thread_message(self, message: "ThreadMessage") -> None:
        # Need to get citations and others and create Pydantic
        pass
        
    async def on_run_step(self, step: "RunStep") -> None:
        logger.info(f"RunStep status: {step.status}")
        if step.status == "failed":
            event = ErrorEvent(
                type="RunStepError",
                message=step.last_error.message,
                code=step.last_error.code
            )
            logger.info(f"A RunStepError occurred. Event: {event}")
            await self.queue.put(event)

        elif step.status == "completed":
            if step.type == "tool_calls":
                for tcall in step.step_details.get("tool_calls", []):
                    step_details_type = tcall.get("type")
                    if step_details_type == "bing_custom_search":
                        request_url = tcall.get("bing_custom_search", {}).get("requesturl", "")
                        if not request_url.strip():
                            return

                        query_str = extract_bing_query(request_url)
                        if not query_str.strip():
                            return

                        event = RunStepEvent(
                            type="RunStep",
                            step_type=step.type,
                            step_details_type=step_details_type,
                            content=query_str
                        )
                        logger.info(f"RunStepEvent: {event}")
                        await self.queue.put(event)


def extract_bing_query(request_url: str) -> str:
    """
    Extract the query string from something like:
      https://api.bing.microsoft.com/v7.0/search?q="latest news about Microsoft January 2025"
    Returns: latest news about Microsoft January 2025
    """
    match = re.search(r'q="([^"]+)"', request_url)
    if match:
        return match.group(1)
    # If no match, fall back to entire request_url
    return request_url




async def on_complete(history: ChatHistory):
    if not history or not history.messages:
        return

    last_msg = history.messages[-1]

    # Try to get citations from metadata
    metadata = getattr(last_msg, "metadata", {}) or {}
    annotations = metadata.get("citations") or metadata.get("annotations") or []

    for item in annotations:
        if "url" in item:
            citations.append({
                "title": item.get("title", "Untitled"),
                "url": item["url"],
                "quote": item.get("quote")
            })


async def format_as_ndjson(r: AsyncGenerator[dict, None]) -> AsyncGenerator[str, None]:
    try:
        async for event in r:
            yield json.dumps(event) + "\n"
    except Exception as e:
        logger.error("An error occurred while formatting NDJSON: %s", e)
        yield json.dumps({"error": str(e)}) + "\n"

    

