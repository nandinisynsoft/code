# PDF → Layout Parsing → Dental X-ray Grounded QA Pipeline

This repository contains a Python reference implementation for the workflow:

1. Parse PDF layout (text/tables/figures) with **LlamaParse**.
2. Detect dental X-rays in extracted figures.
3. Analyze detected X-rays with a **specialized vision model**.
4. Attach findings to nearby text/captions.
5. Embed enriched chunks.
6. Build hybrid retrieval (vector + keyword).
7. Return grounded answers with citations.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

```bash
export LLAMA_CLOUD_API_KEY="..."
export OPENAI_API_KEY="..."
# Optional: custom specialized model endpoint key
export XRAY_MODEL_API_KEY="..."
```

## Usage

```bash
python -m app.pipeline \
  --pdf /path/to/document.pdf \
  --question "What pathology is noted near tooth #19?"
```

### Notes

- The parser is configured to use **OpenAI** as the vendor LLM for parsing.
- If your specialized dental X-ray model is not OpenAI, implement `DentalXRayModelClient` methods for your provider.
- Retrieval returns source chunk IDs/pages for grounding.
