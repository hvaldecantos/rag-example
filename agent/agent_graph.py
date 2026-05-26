import os
from typing import Annotated, NotRequired, Required, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from vector_store import get_vectorstore

load_dotenv()

llm = ChatBedrockConverse(
    model_id=os.getenv("MODEL_ID")
)

vectorstore = get_vectorstore()


class AgentState(TypedDict):
    messages: Required[Annotated[Sequence[BaseMessage], add_messages]]
    token_usage: NotRequired[dict]


@tool(description=os.getenv("RETRIEVER_TOOL_DESCRIPTION"))
def retriever_tool(query: str) -> str:

    # Use similarity_search_with_score to get confidence levels (higher = better match)
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    # print("----------------------------------")
    # for doc, score in docs_with_scores:
    #     print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
    # print("----------------------------------")

    if not docs_with_scores:
        return "I found no relevant information in the available documents."

    results = []
    for doc, distance in docs_with_scores:
        source_file = doc.metadata.get("source_file", "unknown_file.pdf")

        # TODO: PyPDFLoader stores 0-based page index; convert to 1-based for display;
        # check if can use page_label
        page_number = doc.metadata.get("page", -1)
        if isinstance(page_number, int) and page_number >= 0:
            page_label = str(page_number + 1)
        else:
            page_label = "unknown"

        # TODO: Cosine distance is in [0, 2]; 0 means identical, 2 means opposite.
        # Verify if this convertion to confidence percentage is valid.
        confidence = round((1 - (distance / 2)) * 100, 1)

        results.append(
            f"({source_file}, pag. {page_label}, conf. {confidence}%)\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(results)


tools = [retriever_tool]
tools_dict = {tool.name: tool for tool in tools}

llm = llm.bind_tools(tools)


def decision_function(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an AI assistant who answers questions using information loaded into your knowledge base.
Use the retriever_tool available to answer questions grounded in those documents. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Always reference sources using the value of `source_reference` from retrieved chunks.

IMPORTANT: When citing sources in your answer include the name, page, and confidence level; always use this exact format: '(name.pdf, pag. XX, conf. XX%)' that is provided by the retriever_tool.
Every factual claim must be followed by its source citation from the retrieved chunks.
"""


def extract_token_usage(message: BaseMessage) -> dict:
    """Extract token usage from message response_metadata."""
    if not hasattr(message, 'usage_metadata') or not message.usage_metadata:
        return {}
    return {
        'input': message.usage_metadata.get('input_tokens', 0),
        'output': message.usage_metadata.get('output_tokens', 0),
    }


def print_token_summary(state: AgentState) -> None:
    """Print total token usage for the conversation."""
    token_usage = state.get('token_usage', {})
    total_input = token_usage.get('total_input', 0)
    total_output = token_usage.get('total_output', 0)
    total = total_input + total_output

    print(
        f"\n--- TOKEN USAGE SUMMARY: [Input:{total_input} | Output:{total_output} | Total:{total} | Ratio:{total_input/max(total_output, 1):.1f}:1] ---")


def llm_node(state: AgentState) -> AgentState:
    """A simple node for using an LLM to generate a response based on the conversation history."""
    # print("\n=== LLM NODE ===")
    # print(state)
    # print("=====================")
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)

    # Track token usage
    token_usage = state.get(
        'token_usage', {'total_input': 0, 'total_output': 0})
    usage = extract_token_usage(message)
    if usage:
        token_usage['total_input'] += usage['input']
        token_usage['total_output'] += usage['output']
        print(
            f"LLM Call: +{usage['input']} input, +{usage['output']} output tokens")

    return AgentState(messages=[message], token_usage=token_usage)


def retriever_node(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(ToolMessage(
            tool_call_id=t['id'], name=t['name'], content=str(result)))

    # print("Tools Execution Complete. Back to the model!")
    return AgentState(messages=results)


graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_node("retriever", retriever_node)

graph.add_conditional_edges(
    "llm",
    decision_function,
    {True: "retriever", False: END}
)
graph.add_edge("retriever", "llm")
graph.add_edge(START, "llm")


def print_tokens_metrics(state: AgentState) -> None:
    """Print token summary at conversation end."""
    print_token_summary(state)


agent: CompiledStateGraph = graph.compile(checkpointer=MemorySaver())
