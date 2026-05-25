import argparse

from agent_graph import AgentState, agent
from langchain_core.messages import HumanMessage
from utils.graph_image import display_graph
from vector_store import build_vectorstore


def cmd_build(args):
    """Rebuild the vector store embeddings from scratch."""
    print("Rebuilding vector store embeddings...")
    build_vectorstore()
    print("Done.")


def cmd_graph(args):
    """Generate and save an image of the agent graph."""
    output = args.output
    display_graph(agent.get_graph(), filename=output)
    print(f"Graph image saved to {output}")


def cmd_run(args):
    """Start the interactive RAG agent."""
    session_id = args.session_id
    print(f"\n=== RAG AGENT (session: {session_id}) ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("What is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]
        result = agent.invoke(
            AgentState(messages=messages),
            config={"configurable": {"thread_id": session_id}},
        )

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG example – manage embeddings and run the agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Rebuild the vector store embeddings.")

    run_parser = subparsers.add_parser(
        "run", help="Start the interactive RAG agent.")
    run_parser.add_argument(
        "-s", "--session-id",
        default="default",
        help="Session ID used to persist conversation history (default: default).",
    )

    graph_parser = subparsers.add_parser(
        "graph", help="Save an image of the agent graph.")
    graph_parser.add_argument(
        "-o", "--output",
        default="./docs/rag_example_graph.png",
        help="Output file path (default: ./docs/rag_example_graph.png).",
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "graph":
        cmd_graph(args)


if __name__ == "__main__":
    main()
