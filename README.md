# GraphRAG

GraphRAG is a small research/utility project for building graph-assisted RAG (retrieval-augmented generation) workflows. It bundles tools for local embeddings, retrieval, and prompt templates used for graph-based context assembly.

## Contents
- `input/` — source documents (e.g., `book.txt`).
- `cache/` — caches for embeddings and extraction artifacts.
- `Embedding/` — embedding-related configs and helpers (e.g., `litellm_config.yaml`).
- `prompts/` — system and task prompts used by the pipeline.
- `utilities/` — helper modules (`graph_rag_context.py`, `get_litellm_models_name.py`).

## Quick start

1. Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # if provided
```

2. Configure local LLM/embedding server (see below).

3. Run your scripts or notebooks (e.g., open `GraphRAG query modes test.ipynb`).

## Local embedding server (lamafile + litellm)

This project supports running a local embedding server backed by `lamafile` and `litellm` for offline embeddings and lightweight LLM usage.

Basic idea:
- `lamafile` serves embedding models and lightweight model files locally.
- `litellm` provides a small HTTP API wrapper to host embedding/LLM endpoints.

Model file reference:

- Download the `mxbai-embed` llamafile from Hugging Face and place it in the `Embedding/` folder:

    https://huggingface.co/mozilla-ai/mxbai-embed-large-v1-llamafile/blob/main/mxbai-embed-large-v1-f16.llamafile

Recommended minimal steps:

1. Install `litellm` (or your local server of choice) and ensure it's on PATH or available in the environment.

2. Start a local server exposing embedding and/or LLM endpoints. Example (pseudo-command):

```powershell
# start litellm server reading local lamafile models
litellm serve --config Embedding/litellm_config.yaml --port 8080
```

3. Point GraphRAG embedding code to the local server URL (e.g., `http://localhost:8080/embed`). Update `Embedding/litellm_config.yaml` as needed.

Notes:
- The project doesn't bundle proprietary models. Use locally-available models in `lamafile` or other local model stores.
- For production or GPU usage, use the recommended provider instructions for `litellm` or your preferred model server.

## GraphRAG retriever API example

The project includes retriever-based access patterns. Below is an example Python snippet showing how to call a GraphRAG-style retriever API while controlling retrieval parameters and optionally bypassing any LLM call.

```python
from utilities.graph_rag_context import GraphRAGRetriever

# create retriever (example API surface)
retriever = GraphRAGRetriever(
    endpoint="http://localhost:8000/retriever",
    index_path="lancedb/",
)

# retrieval parameters you can control
params = {
    "k": 10,                    # number of candidates
    "filter": None,             # optional metadata filter
    "score_threshold": 0.2,     # minimum score
    "use_graph": True,          # whether to include graph context
}

# bypass the LLM: retrieve context only
results = retriever.retrieve(query="What is the main claim?", params=params, bypass_llm=True)

# normal flow: retrieve + LLM step
response = retriever.retrieve(query="Summarize the claims.", params=params, bypass_llm=False)

print(results)
print(response)
```

Key parameters and behaviors to consider:
- `k` (int): number of nearest neighbors to include.
- `filter` (dict|None): optional metadata filters for retrieval.
- `use_graph` (bool): whether to incorporate graph-expanded context.
- `bypass_llm` (bool): if true, return retrieved context only and skip calling any LLM — useful for debugging or when you want to do offline downstream processing.
- `control` (dict): free-form control object you can pass to tweak reranking, graph traversal depth, or prompt selection.

Example `control` usage:

```python
control = {"rerank_model": "cross-encoder-small", "graph_depth": 2}
response = retriever.retrieve(query, params=params, control=control)
```

## Configuration and environment notes

- `settings.yaml` contains project-level settings used by scripts.
- `Embedding/litellm_config.yaml` holds local server settings for `litellm`.
- Keep large caches and generated embeddings out of version control: see `.gitignore`.

## Helpful commands

Start a lightweight HTTP embedding test (example):

```powershell
curl -X POST http://localhost:8080/embed -d "{\"texts\":[\"hello world\"]}"
```

Run a notebook:

```powershell
jupyter notebook "GraphRAG query modes test.ipynb"
```

## Local server commands

The following commands are taken from `Embedding/reference/cmd.txt` and show example local-server startup and project initialization steps.

```powershell
# llamafile
conda activate your_env
cd Embedding
.\mxbai-embed-large-v1-f16.llamafile.exe --server --nobrowser --embedding --port 8080 -ngl 9999

# litellm
conda activate your_env
cd Embedding
litellm --config litellm_config.yaml

# GraphRAG - init
conda activate your_env
cd .
graphrag index --method fast
```

## Contributing & notes

- Add new embeddings under `Embedding/` and update `Embedding/litellm_config.yaml`.
- Put prompt files under `prompts/` and reference them from your pipelines.
- Use `utilities/` helpers for common operations; extend them if needed.

---
Created files:
- README.md (this file)
