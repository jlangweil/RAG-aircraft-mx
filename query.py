import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

INDEX_DIR = Path("output/faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are an expert aviation maintenance analyst with deep knowledge 
of FAA regulations, aircraft systems, and maintenance records.

You will be given excerpts from aircraft maintenance work orders and asked questions 
about them. Answer based ONLY on the provided context — do not invent or assume 
information not present in the records.

For each answer:
- Be specific and technical where the records support it
- Always cite which work order, aircraft registration, and date the information 
  comes from
- If the records don't contain enough information to answer, say so clearly
- Flag any safety-relevant findings (AD compliance, recurring squawks, items 
  noted as due soon)"""


def load_vectorstore() -> FAISS:
    """Load the FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_chunks(vectorstore: FAISS, question: str, k: int = 5) -> list:
    """Retrieve the k most relevant chunks for a question."""
    return vectorstore.similarity_search(question, k=k)


def format_context(chunks: list) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk.metadata
        header = (
            f"[Source {i}] "
            f"Aircraft: {m.get('aircraft_registration', 'Unknown')} | "
            f"Work Order: {m.get('work_order', 'Unknown')} | "
            f"Date: {m.get('date', 'Unknown')} | "
            f"Tach: {m.get('tach_time', 'Unknown')}"
        )
        context_parts.append(f"{header}\n{chunk.page_content}")

    return "\n\n---\n\n".join(context_parts)


def ask(question: str, vectorstore: FAISS) -> dict:
    """Full RAG pipeline: retrieve chunks, build prompt, get answer."""
    chunks = retrieve_chunks(vectorstore, question)
    context = format_context(chunks)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Based on the following maintenance records, please answer this question:

Question: {question}

Maintenance Records:
{context}"""
            }
        ]
    )

    return {
        "answer": response.content[0].text,
        "sources": chunks,
        "context": context
    }