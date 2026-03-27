from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from chunker import load_and_chunk

CLEANED_DIR = Path("output/cleaned_text")
INDEX_DIR = Path("output/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# This model is small, fast, and very good for semantic similarity
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunks_to_documents(cleaned_dir: Path) -> list[Document]:
    """Convert all cleaned text files into LangChain Documents."""
    documents = []
    txt_files = list(cleaned_dir.glob("*.txt"))

    if not txt_files:
        print(f"No cleaned text files found in {cleaned_dir}")
        return documents

    for txt_path in txt_files:
        print(f"Chunking: {txt_path.name}")
        chunks = load_and_chunk(txt_path)
        print(f"  → {len(chunks)} chunks")

        for chunk in chunks:
            # LangChain Document = text content + metadata dict
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "source_file":            chunk.source_file,
                    "aircraft_registration":  chunk.aircraft_registration,
                    "aircraft_serial":        chunk.aircraft_serial,
                    "work_order":             chunk.work_order,
                    "date":                   chunk.date,
                    "tach_time":              chunk.tach_time,
                    "chunk_type":             chunk.chunk_type,
                    "item_number":            chunk.item_number,
                }
            )
            documents.append(doc)

    return documents


def build_index(documents: list[Document]):
    """Embed documents and save FAISS index to disk."""
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run will download ~90MB — one time only)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  #change to cuda if needed
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Embedding {len(documents)} chunks...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(str(INDEX_DIR))
    print(f"\n✓ FAISS index saved to {INDEX_DIR}")
    print(f"  {len(documents)} chunks indexed and ready to query")


if __name__ == "__main__":
    docs = chunks_to_documents(CLEANED_DIR)
    if docs:
        build_index(docs)