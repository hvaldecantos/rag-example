from agent_graph import AgentState, agent
from utils.graph_image import display_graph
from langchain_core.messages import HumanMessage
from agent.vector_store import get_vectorstore

filename = "docs/rag_example_grph.png"
display_graph(agent.get_graph(), filename=filename)

vector_store = get_vectorstore()

result = agent.invoke(AgentState(
    messages=[HumanMessage(content="hello agent!")]))

print(result)
