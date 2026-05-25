from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from utils.graph_image import display_graph

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


llm = ChatBedrockConverse(
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
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

agent = graph.compile()

filename = "docs/rag_example_grph.png"

display_graph(agent.get_graph(), filename=filename)

result = agent.invoke(AgentState(
    messages=[HumanMessage(content="hello agent!")]))

print(result)
