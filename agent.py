
import logging, json
import re
from typing import AsyncGenerator, Any, Union
from pydantic import BaseModel
from fastapi import HTTPException
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import Agent, AgentThread, MessageRole
from azure.ai.projects.models import ThreadMessage

from models import AgentRequest, MessageEvent, ErrorEvent, Citation, CitationsEvent, CreateThreadEvent, DeleteThreadRequest
logger = logging.getLogger(__name__)


async def stream_agent_response(agent_request: AgentRequest, project_client: AIProjectClient):
    thread_id = agent_request.thread_id
    logger.info(f"Running Thread ID: {thread_id}")
    logger.info(f"Running Agent ID: {agent_request.agent_id}")
    if thread_id is None:
        thread = await project_client.agents.create_thread()
        logger.info(f"Created new thread ID: {thread.id}")
        thread_id = thread.id
    else:
        logger.info(f"Retrieved thread ID: {thread_id}")

    create_thread = CreateThreadEvent(type="create_thread", thread_id=thread_id)
    yield create_thread

    message = await project_client.agents.create_message(
        thread_id=thread_id,
        role=MessageRole.USER,
        content=agent_request.message,
    )

    run = await project_client.agents.create_and_process_run(
        thread_id=thread_id, 
        agent_id=agent_request.agent_id
    )
    if run.status == "failed":
        event = ErrorEvent(
            type="error", 
            message=run.last_error.message,
            code=run.last_error.code,)
        logger.error(f"An Error occurred. Event: {event}")
        yield event
        return
    
    messages = await project_client.agents.list_messages(thread_id=thread_id)
    response_message = messages.get_last_message_by_role(MessageRole.AGENT)

    if not response_message:
        logger.error("No response message found.")
        return

    citations = []
    for text_message in response_message.text_messages:
        event = MessageEvent(
            type=text_message.type,
            role=MessageRole.AGENT,
            message=text_message.text.value,
        )
        logger.info(f"Message Event: {event}")
        yield event

    for annotation in response_message.url_citation_annotations:
        citation = Citation(
            type=annotation.type,
            title=annotation.url_citation.title,
            url=annotation.url_citation.url,
            start_index=annotation.start_index,
            end_index=annotation.end_index,
        )
        logging.info(f"Citation: {citation}")
        citations.append(citation)

    yield CitationsEvent(
        type="citations_event",
        citations=citations,
    )

    

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

async def delete_thread(request: DeleteThreadRequest, project_client: AIProjectClient):
    """
    Delete a thread by ID.
    """
    thread_id = request.thread_id
    try:
        await project_client.agents.delete_thread(thread_id=thread_id)
        logging.info(f"Thread {thread_id} deleted successfully.")
        return {"status": "success", "message": f"Thread {thread_id} deleted."}
    except Exception as e:
        logger.error("Thread deletion failed for %s: %s", thread_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to delete thread {thread_id}: {str(e)}")

async def format_as_ndjson(r: AsyncGenerator[Union[BaseModel, dict, list, str], None]) -> AsyncGenerator[str, None]:
    try:
        async for event in r:
            if isinstance(event, BaseModel):
                # v2: model_dump();  v1: dict()
                logger.debug(f"Event: {event}")
                event = event.model_dump() if hasattr(event, "model_dump") else event.dict()

            yield json.dumps(event, ensure_ascii=False) + "\n"

    except Exception as e:
        logger.error("An error occurred while formatting NDJSON: %s", e)
        yield json.dumps({"error": str(e)}) + "\n"

    