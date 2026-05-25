from typing import TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # a reducer function
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from utils.graph_image import display_graph


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def llm_node(state: AgentState) -> AgentState:
    """A simple node that mimics an AI response."""
    message = AIMessage(content="hello from node!")
    return {'messages': [message]}


graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

agent = graph.compile()

filename = "docs/rag_example_grph.png"

display_graph(agent.get_graph(), filename=filename)

result = agent.invoke(AgentState(
    messages=[HumanMessage(content="hello node!")]))

print(result)
