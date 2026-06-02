# Service Order RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for analyzing equipment service order history stored in CSV files.

The chatbot uses:

* FAISS for vector search
* NVIDIA NIM Embeddings (or local embeddings)
* NVIDIA NIM LLMs (or local LLMs)
* LangChain
* Python

The system converts service order records into vector embeddings, stores them in a FAISS index, and retrieves the most relevant records when users ask questions.

---

## Features

* Search thousands of historical service orders
* Natural language questions
* Vector similarity search using FAISS
* NVIDIA NIM embedding support
* NVIDIA NIM chat model support
* Local embedding model support
* Real-time indexing progress
* Source-aware answers from service order history

---

## Example Questions

* Which jobs mention hydraulic issues?
* Show all 1000 hour services.
* What is the highest labor hour repair?
* Find ECU fault related work orders.
* What repairs are similar to boom drift issues?
* Which service orders involve engine diagnostics?

---

## Project Structure

```text
ChatBot/
│
├── serviceorder.csv
├── ingest.py
├── chat.py
├── requirements.txt
├── .gitignore
├── .env
│
└── serviceorder_index/
    ├── index.faiss
    └── index.pkl
```

---

## Installation

### Clone Repository

```bash
git clone <your-repository-url>
cd ChatBot
```

### Create Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file.

```env
NVIDIA_NIM_API_KEY=your_api_key

BASE_URL=https://integrate.api.nvidia.com/v1

EMBEDDING_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2

CHAT_MODEL=meta/llama-3.3-70b-instruct
```

---

## Input Data

Place your service order data in:

```text
serviceorder.csv
```

Example:

```csv
Make,Model,Meter,ServiceOrderID,Wo Number,Wo Segment,Description,Contract Type,M_Actual_ServiceOrderLaborHrsTotal,Wo Status
JD,470P,2824,S100064029140174003,6402914,3,647.16,C&F,1,I
JD,470P,1089,S100047118260157004,4711826,4,All Hydraulic Pressures set to Top of Specification,C&F,2.25,I
```

---

## Build the Vector Index

Run:

```bash
python ingest.py
```

Expected output:

```text
Loading CSV...
Rows Loaded: 4527

Creating embeddings...

Completed: 25/4527 (0.55%)
Completed: 50/4527 (1.10%)

...

INDEX CREATION COMPLETE
```

This creates:

```text
serviceorder_index/
```

containing the FAISS vector database.

---

## Start the Chatbot

Run:

```bash
python chat.py
```

Example:

```text
Question:
Which service orders mention hydraulic issues?
```

Response:

```text
Work Order 4711826 mentions hydraulic pressure adjustments.

Description:
All Hydraulic Pressures set to Top of Specification

Labor Hours:
2.25
```

---

## Using Local Embeddings (Optional)

Instead of NVIDIA embeddings, install:

```bash
pip install sentence-transformers
pip install langchain-huggingface
```

Use:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```

Benefits:

* No API costs
* No rate limits
* Faster indexing for small datasets
* Fully local processing

---

## Recommended Architecture

```text
User Question
      │
      ▼
FAISS Similarity Search
      │
      ▼
Relevant Service Orders
      │
      ▼
NVIDIA NIM LLM
      │
      ▼
Answer
```

---

## Future Improvements

* Streamlit Web UI
* Chat history and memory
* Multiple CSV ingestion
* Service Advisor integration
* Power BI integration
* Source citations
* Hybrid keyword + vector search
* PostgreSQL / pgvector backend

---

## Security Notes

Do not commit:

* `.env`
* API keys
* FAISS indexes
* Production datasets

Use `.gitignore` to exclude sensitive files.

---

## License

MIT License
