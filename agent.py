
import logging, json
import re
from typing import AsyncGenerator, Any, Union
from pydantic import BaseModel
from fastapi import HTTPException
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import Agent, AgentThread, MessageRole
from azure.ai.projects.models import ThreadMessage

from models import AgentRequest, MessageEvent, ErrorEvent, Citation, CitationsEvent, CreateThreadEvent, DeleteThreadRequest 
from models import BingGroundingEvent, FileSearchItem, FileSearchEvent
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

    for text_message in response_message.text_messages:
        event = MessageEvent(
            type=text_message.type,
            role=MessageRole.AGENT,
            message=text_message.text.value,
        )
        logger.info(f"Message Event: {event}")
        yield event

    citations = []
    run_step_list = await project_client.agents.list_run_steps(thread_id=thread_id, run_id=run.id, include=["step_details.tool_calls[*].file_search.results[*].content"])
    for run_step in run_step_list.data:
        if run_step.status == "failed":
            event = ErrorEvent(
                type="RunStepError",
                message=run_step.last_error.message,
                code=run_step.last_error.code
            )
            logger.error(f"A RunStepError occurred. Event: {event}")
            yield event
        elif run_step.type == "tool_calls":
            for tcall in run_step.step_details.get("tool_calls", []):
                step_details_type = tcall.get("type")
                if step_details_type == "bing_grounding":
                    request_url = tcall.get("bing_grounding", {}).get("requesturl", "")
                    if not request_url.strip():
                        return

                    query_str = extract_bing_query(request_url)
                    if not query_str.strip():
                        return

                    event = BingGroundingEvent(
                        type=step_details_type,
                        title=f"AI Web Search: {query_str}",
                        url=f"https://www.bing.com/search?q={query_str}",
                    )
                    logging.info(f"BingGroundingEvent: {event}")
                    yield event

                elif step_details_type == "file_search":
                    file_res = tcall.get("file_search", {}).get("results", [])
                    items = []
                    for f in file_res:
                        content_arr = f.get("content", [])
                        content_item = content_arr[0] if len(content_arr) > 0 else {}
                        item = FileSearchItem(
                            type=step_details_type, 
                            name=f.get("file_name", ""),
                            content=content_item.get("text", ""),
                        )
                        items.append(item)
                    if items:
                        yield FileSearchEvent(
                            type="file_search_event",
                            items=items,
                        )
                    

    for annotation in response_message.url_citation_annotations:
        citation = Citation(
            type=annotation.type,
            title=annotation.url_citation.title,
            url=annotation.url_citation.url,
            start_index=annotation.start_index,
            end_index=annotation.end_index,
        )
        logging.info(f"Web Citation: {citation}")
        citations.append(citation)

    if citations:
        yield CitationsEvent(
            type="citations_event",
            citations=citations,
        )



def extract_bing_query(request_url: str) -> str:
    """
    Extract the query string from something like:
      https://api.bing.microsoft.com/v7.0/search?q="latest news about Microsoft January 2025"
    Returns: latest news about Microsoft January 2025
    """
    match = re.search(r'[?&]q=([^&]+)', request_url)
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

    