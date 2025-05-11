import logging, json
from typing import AsyncGenerator
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread, Agent, AgentResponseItem
from semantic_kernel.contents import ChatMessageContent
from azure.ai.projects import AIProjectClient

from pydantic import ValidationError
from models import AgentRequest, ThreadInitEvent, MessageStreamEvent, ErrorStreamEvent

logger = logging.getLogger(__name__)

async def stream_agent_response(agent_request: AgentRequest, client: AIProjectClient, agent: AzureAIAgent):
    thread: AzureAIAgentThread = None
    response_item: AgentResponseItem = None
    if agent_request.thread_id == -1:
        thread = AzureAIAgentThread(client=client)
        # Need to save and send thread_id to front end
    else:
        thread = AzureAIAgentThread(client=client, thread_id=agent_request.thread_id)
    
    thread_id = thread.id

    init_event = ThreadInitEvent(type="thread_init", thread_id=thread_id)
    yield init_event



    #### MOVE TO MAIN
    bing_custom_connection = client.connections.get(connection_name=os.environ["BING_CUSTOM_CONNECTION_NAME"])
    conn_id = bing_custom_connection.id

    print(conn_id)

    # Initialize agent bing custom search tool and add the connection id
    bing_custom_tool = BingCustomSearchTool(connection_id=conn_id, instance_name="<config_instance_name>")

    ### ALSO FIGURE OUT AzureAIAgentSettings
    
    async for response_item in agent.invoke_stream(
        messages=agent_request.message,
        thread=thread, 
        tools=bing_custom_tool.definitions
    ):
        try:
            content = response_item.content

            if isinstance(content, ChatMessageContent) and content.content:
                event = MessageStreamEvent(
                    type="message",
                    role=content.role,
                    message=content.content
                )
                yield event
        
        except (AttributeError, TypeError, ValidationError) as e:
            error_event = ErrorStreamEvent(
                type="error",
                message="An error occurred while streaming the agent response.",
                details=str(e),
                code="streaming_error"
            )
            yield error_event
            break
    
    # Either get last message or callback function to get citations and yield pydantic model
    



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

    

