# Guardrail RAG

This project implements a retrieval-augmented generation flow with:

- a safety guardrail step
- hybrid retrieval using vector similarity and BM25-style scoring
- RRF reranking
- LLM-based answer generation

## Configuration

Copy [.env.example](.env.example) to .env and fill in the values for your provider.

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
```

### Azure OpenAI

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
```

If no API key is configured, the app falls back to a lightweight non-LLM response path.

## Run tests

```bash
python -m pytest -q
```
