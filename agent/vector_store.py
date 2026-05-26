import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY")
DOCUMENTS_DIRECTORY = os.getenv("DOCUMENTS_DIRECTORY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0"
)


def _find_pdf_paths() -> list[str]:
    """Return all PDF files under DOCUMENTS_DIRECTORY (recursive)."""
    root = Path(DOCUMENTS_DIRECTORY)
    if not root.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIRECTORY}")

    pdf_paths = sorted(str(path) for path in root.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found under: {DOCUMENTS_DIRECTORY}")

    return pdf_paths


def build_vectorstore() -> Chroma:
    """Load PDFs, chunk them, embed them, and persist to ChromaDB."""
    pdf_paths = _find_pdf_paths()
    pages = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        loaded_pages = loader.load()

        # Keep lightweight citation fields in metadata for answer attribution.
        source_name = os.path.basename(pdf_path)
        for page in loaded_pages:
            page.metadata["source_file"] = source_name
            page.metadata["source_path"] = pdf_path

        pages.extend(loaded_pages)
        print(f"Loaded {len(loaded_pages)} pages from {source_name}")

    print(f"Total loaded pages: {len(pages)} from {len(pdf_paths)} PDFs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Build a clean collection on every explicit rebuild to avoid stale chunks.
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(
        f"Vector store built and persisted ({len(chunks)} chunks from {len(pdf_paths)} PDFs).")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB collection without re-embedding."""
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def _collection_exists() -> bool:
    """Return True if the ChromaDB collection already has documents."""
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def get_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    Return a Chroma vector store.
    - If force_rebuild is True, always re-embed and overwrite.
    - Otherwise, load the existing collection when it exists; build it on first run.
    """
    if force_rebuild or not _collection_exists():
        action = "Rebuilding" if force_rebuild else "No existing collection found — building"
        print(f"{action} vector store...")
        return build_vectorstore()

    return load_vectorstore()
