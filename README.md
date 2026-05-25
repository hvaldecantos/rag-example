# A RAG Example

A small AI system using Retrieval-Augmented Generation (RAG). The agent answers questions grounded in a set of PDF documents, citing sources with page numbers and confidence scores.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS credentials configured for Amazon Bedrock

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the project root with the required configuration:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...

MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
EMBEDDINGS_MODEL_ID="amazon.titan-embed-text-v2:0"
PERSIST_DIRECTORY = "./embeddings/"
DOCUMENTS_DIRECTORY = "./embeddings/documents"
DEFAULT_COLLECTION_NAME = "general"
```

Then build the vector store by indexing all PDFs in `DOCUMENTS_DIRECTORY`:

```bash
uv run agent/main.py build
```

This embeds all documents into the local ChromaDB store. Re-run `build` whenever you add or update documents.

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_ID` | AWS Bedrock model ID for LLM and embeddings | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `EMBEDDINGS_MODEL_ID` | AWS Bedrock model ID for LLM and embeddings | `amazon.titan-embed-text-v2:0` |
| `DOCUMENTS_DIRECTORY` | Path to folder containing PDF files (searched recursively) | `./embeddings/documents` |
| `PERSIST_DIRECTORY` | Where to store ChromaDB embeddings | `./embeddings` |
| `DEFAULT_COLLECTION_NAME` | ChromaDB collection name | `general` |

## Running the Agent

Start an interactive session:

```bash
uv run agent/main.py run
```

The agent keeps full conversation history within a session, so you can ask follow-up questions that reference previous answers.

### Session ID

Use `--session-id` (or `-s`) to name a session. Each unique ID has its own isolated conversation history:

```bash
uv run agent/main.py run --session-id my-session
```

Omitting the flag uses the `default` session.

### Commands

| Command | Description |
|---------|-------------|
| `build` | Index PDF documents into the vector store |
| `run` | Start the interactive agent |
| `run -s <id>` | Start the agent with a named session |
| `graph` | Save a diagram of the agent graph |

#### `graph` options

```bash
uv run agent/main.py graph --output ./docs/my_graph.png
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `./docs/rag_example_graph.png` | Output file path for the graph image |
