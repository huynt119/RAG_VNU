# RAG_VNU

## Environment
This project requires the `OPENROUTER_API_KEY` environment variable to access the API. To manage environment variables more easily, it's recommended to use a `.env` file along with the `python-dotenv` library.

### ✅ Step 1: Install `python-dotenv`

First, activate your virtual environment if you have one, then install the library:

```bash
pip install python-dotenv
```

### ✅ Step 2: Create a file named `.env` in the root directory

```bash
OPENROUTER_API_KEY=your_actual_api_key_here
```

Replace your_actual_api_key_here with your actual API key. Get your own API key in [link](https://openrouter.ai/settings/keys)

## 🧠 Project Overview

This project contains several scripts and notebooks for processing, retrieving, and evaluating information. Below is a breakdown of each file and its purpose:

### 📂 Source Files

| File Name                 | Description                              |
|---------------------------|------------------------------------------|
| `batch_rag_eval.py`       | Evaluate RAG on test data                |
| `chunking_cluster.py`     | Chunk clean data                         |
| `clean_data.py`           | Clean data                               |
| `crawl_with_thread.ipynb` | Multi-thread crawl data                  |
| `IAA.ipynb`               | Evaluate IAA on test data                |
| `multi_hop_new.py`        | Multi-hop RAG system                     |
| `rag_eval.py`             | Metric to evaluate RAG system            |
| `retriever.py`            | Retriever system                         |
| `vector_store.py`         | Build vector database                    |
