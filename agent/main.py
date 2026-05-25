from agent_graph import AgentState, agent
from langchain_core.messages import HumanMessage
from utils.graph_image import display_graph

display_graph(agent.get_graph(), filename="docs/rag_example_graph.png")

result = agent.invoke(AgentState(
    messages=[HumanMessage(content="What was the water consumption of AWS in 2021?")]))

print(result)
