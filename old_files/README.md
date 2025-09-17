# A2A RAG Agent

1. Create a virtual environment

```
uv venv
source .venv/bin/activate
```

2. Install dependencies

```
uv sync
```

## Deploy with Ollama

## Deploy LLM Locally with Ollama

1. Install Ollama

2. Install local model

```
ollama pull gemma3:1b-it-qat
```

3. List model

```
ollama list
```

4. Run the server

```
ollama serve
```

5. Remove the model from ollama

```
ollama rm gemma3:4b
```
