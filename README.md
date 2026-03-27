# ✈️ Aircraft Maintenance Records RAG

A Retrieval-Augmented Generation (RAG) system for querying aircraft maintenance records using natural language. Built for flying clubs managing maintenance logs across multiple aircraft or anyone purchasing a new aircraft to review logs.

Ask questions like:
- *"Has N8050J had any magneto issues?"*
- *"What ADs have been complied with?"*
- *"Were there any fuel system problems found?"*
- *"What was the labor breakdown on the last annual?"*

---

## Project Structure

```
mx-rag/
├── app.py                  # Streamlit UI
├── query.py                # RAG retrieval and answer generation
├── ingest.py               # Chunk, embed, and build FAISS index
├── chunker.py              # Smart work-item-aware chunking
├── extract.py              # AWS Textract extraction (alternative pipeline)
├── cleanup.py              # LLM-based OCR cleanup (Textract pipeline)
├── .env.template           # Environment variable template
├── requirements.txt        # Python dependencies
└── data/
    └── raw_pdfs/           # Drop your maintenance PDFs here
```

---

## Prerequisites

- Python 3.12+
- An Anthropic API key (required)
- AWS credentials (needed for Textract extraction pipeline)

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/jlangweil/mx-rag.git
cd mx-rag
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.template .env
```

Edit `.env` and fill in your API keys (see setup sections below).

### 5. Add your maintenance PDFs

Copy your scanned maintenance work order PDFs into `data/raw_pdfs/`.

### 6. Extract text from PDFs

```bash
python extract.py
```

This uses Claude vision to extract text from scanned/handwritten PDFs. Output lands in `output/vision_extract/`.

### 7. Build the vector index

```bash
python ingest.py
```

This chunks the extracted text by work item, generates embeddings, and builds the FAISS index. Output lands in `output/faiss_index/`.

### 8. Run the app

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`.

---

## Getting an Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Navigate to **API Keys** in the left sidebar
4. Click **Create Key**
5. Give it a name (e.g. `mx-rag`)
6. Copy the key — it starts with `sk-ant-`
7. Add it to your `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

> **Note:** Keep your API key secret. Never commit it to git. The `.env` file is excluded by `.gitignore`.

---

## AWS Setup (Textract Pipeline Only)

The Textract pipeline (`extract.py`) requires an AWS account and IAM credentials.

### Step 1 — Create an AWS account

If you don't have one, sign up at [aws.amazon.com](https://aws.amazon.com). New accounts include a free tier that covers this project's usage.

### Step 2 — Create an IAM user for Textract

It is best practice to create a dedicated IAM user with minimal permissions rather than using your root account.

1. Go to the [IAM Console](https://console.aws.amazon.com/iam)
2. Click **Users** → **Create user**
3. Enter a username (e.g. `mx-rag-textract`)
4. Select **Attach policies directly**
5. Search for and attach **AmazonTextractFullAccess**
6. Click through to **Create user**

### Step 3 — Create an access key

1. Click on your newly created user
2. Go to the **Security credentials** tab
3. Scroll to **Access keys** → **Create access key**
4. Select **Local code** as the use case
5. Copy both the **Access Key ID** and **Secret Access Key** — you will not be able to see the secret again

### Step 4 — Install and configure the AWS CLI

```bash
# Install AWS CLI (if not already installed)
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

aws configure
```

Enter your credentials when prompted:

```
AWS Access Key ID:      your-access-key-id
AWS Secret Access Key:  your-secret-access-key
Default region name:    us-east-1
Default output format:  json
```

This writes credentials to `~/.aws/credentials` which boto3 reads automatically.

### Step 5 — Add to .env (optional)

Alternatively, you can store AWS credentials directly in `.env`:

```
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
AWS_REGION=us-east-1
```

### Textract pricing

- **$0.0015 per page** for basic text and handwriting extraction
- Free tier: 1,000 pages/month for the first 3 months
- A typical flying club corpus of 500 pages costs approximately **$0.75**

---

## Architecture Notes

### Why RAG instead of just asking Claude?

Claude's training data doesn't include your club's maintenance records. RAG grounds the answers in your actual documents — Claude only sees the retrieved chunks, so every answer is traceable back to a specific work order.

### Why chunk by work item?

Maintenance work orders contain numbered items (1, 2, 3...) each describing a discrete repair action. Chunking by work item rather than fixed character count keeps each maintenance action as a complete semantic unit, improving retrieval accuracy.

### Why FAISS?

FAISS runs entirely locally — no cloud account, no API key, no cost. The index is built once and reused. For a production deployment with multiple users, FAISS can be swapped for Pinecone or Chroma with minimal code changes (one import and two function calls).

### Swapping to a hosted vector database

The project is architected for easy migration to a cloud vector store. In `ingest.py` and `query.py`, replace the FAISS lines with:

```python
# pip install langchain-pinecone
from langchain_pinecone import PineconeVectorStore
```

This enables deployment to Streamlit Community Cloud or other hosted platforms.

---

## Dependencies

See `requirements.txt` for full list. Key packages:

- `anthropic` — Claude API client
- `langchain` / `langchain-community` / `langchain-huggingface` — RAG framework
- `faiss-cpu` — local vector store
- `sentence-transformers` — HuggingFace embeddings
- `pymupdf` — PDF to image conversion (no external dependencies)
- `streamlit` — web UI
- `boto3` — AWS SDK (Textract pipeline only)

