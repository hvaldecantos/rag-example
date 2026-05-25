import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatBedrockConverse(
    model_id=os.getenv("MODEL_ID")
)

system_prompt = """
You are an AI assistant. Please answer my query using your capabilities.
"""


def llm_node(state: AgentState) -> AgentState:
    """A simple node for using an LLM to generate a response based on the conversation history."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

agent: CompiledStateGraph = graph.compile()
