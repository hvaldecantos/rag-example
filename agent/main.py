from agent_graph import AgentState, agent
from langchain_core.messages import HumanMessage
from utils.graph_image import display_graph
from vector_store import build_vectorstore

display_graph(agent.get_graph(), filename="docs/rag_example_graph.png")

# build_vectorstore()


def start_agent():
    print("\n=== RAG AGENT===")

    while True:

        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        messages = [HumanMessage(content=user_input)]
        result = agent.invoke(AgentState(messages=messages))

        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


if __name__ == "__main__":
    start_agent()
