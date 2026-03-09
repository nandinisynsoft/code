from __future__ import annotations

import argparse
import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ParsedFigure:
    id: str
    page: int
    image_path: Path | None = None
    caption: str | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ParsedChunk:
    id: str
    page: int
    text: str
    kind: str = "text"  # text | table | figure_context
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class XRayFinding:
    figure_id: str
    page: int
    summary: str
    structured: dict[str, Any]


class DentalXRayModelClient:
    """Specialized model client for dental X-ray detection and analysis.

    Default implementation uses OpenAI vision-capable models; swap this class if
    you have a dedicated radiology model/API.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1") -> None:
        self.api_key = api_key or os.getenv("XRAY_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model

    def _openai_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Please install openai package.") from exc
        if not self.api_key:
            raise RuntimeError("XRAY_MODEL_API_KEY or OPENAI_API_KEY is required.")
        return OpenAI(api_key=self.api_key)

    @staticmethod
    def _img_to_data_url(image_path: Path) -> str:
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def detect_xray(self, image_path: Path) -> bool:
        """Returns True if image appears to be a dental X-ray/radiograph."""
        client = self._openai_client()
        data_url = self._img_to_data_url(image_path)
        resp = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is this a dental X-ray/radiograph? Answer only JSON: {\"is_dental_xray\": true/false}"},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            max_output_tokens=80,
        )
        text = getattr(resp, "output_text", "") or ""
        try:
            result = json.loads(_extract_json(text))
            return bool(result.get("is_dental_xray", False))
        except Exception:
            return "true" in text.lower()

    def analyze_xray(self, image_path: Path) -> XRayFinding:
        raise NotImplementedError("Use analyze_xray_for_figure with figure metadata.")

    def analyze_xray_for_figure(self, figure: ParsedFigure) -> XRayFinding:
        if figure.image_path is None:
            raise ValueError(f"Figure {figure.id} has no image path.")

        client = self._openai_client()
        data_url = self._img_to_data_url(figure.image_path)

        prompt = (
            "You are a dental radiology assistant. Analyze this dental X-ray and return JSON with keys: "
            "findings (array of strings), likely_conditions (array), tooth_regions (array), confidence (0-1), impression (string)."
        )
        resp = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            max_output_tokens=600,
        )
        text = getattr(resp, "output_text", "") or "{}"
        try:
            structured = json.loads(_extract_json(text))
        except Exception:
            structured = {"raw": text}

        summary = structured.get("impression") if isinstance(structured, dict) else None
        summary = summary or "Dental X-ray analyzed; see structured payload."
        return XRayFinding(figure_id=figure.id, page=figure.page, summary=summary, structured=structured)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Please install openai package.") from exc
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
        return OpenAI(api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._client()
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class HybridRetriever:
    def __init__(self, chunks: list[ParsedChunk], embedder: OpenAIEmbedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.chunk_vectors = embedder.embed([c.text for c in chunks])

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = sum(x * x for x in a) ** 0.5
        db = sum(y * y for y in b) ** 0.5
        return (num / (da * db)) if da and db else 0.0

    @staticmethod
    def _tokenize(s: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", s.lower())

    def _bm25_like(self, query: str) -> list[float]:
        q = set(self._tokenize(query))
        scores: list[float] = []
        for c in self.chunks:
            toks = self._tokenize(c.text)
            overlap = sum(1 for t in toks if t in q)
            scores.append(float(overlap) / max(len(toks), 1))
        return scores

    def search(self, query: str, k: int = 5, alpha: float = 0.7) -> list[tuple[ParsedChunk, float]]:
        qvec = self.embedder.embed([query])[0]
        vec_scores = [self._cosine(qvec, cvec) for cvec in self.chunk_vectors]
        kw_scores = self._bm25_like(query)
        scored = []
        for i, c in enumerate(self.chunks):
            score = alpha * vec_scores[i] + (1 - alpha) * kw_scores[i]
            scored.append((c, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def parse_pdf_with_llamaparse(pdf_path: Path) -> tuple[list[ParsedChunk], list[ParsedFigure]]:
    """Parse PDF with LlamaParse and OpenAI vendor model for layout-aware extraction."""
    try:
        from llama_parse import LlamaParse
    except ImportError as exc:
        raise RuntimeError("Please install llama-parse package.") from exc

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY is required.")

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=True,
        language="en",
        vendor_multimodal_model_name="openai-gpt-4.1-mini",
        vendor_multimodal_api_key=openai_key,
    )

    docs = parser.load_data(str(pdf_path))

    chunks: list[ParsedChunk] = []
    figures: list[ParsedFigure] = []

    for i, d in enumerate(docs, start=1):
        md = getattr(d, "text", "")
        page = int((getattr(d, "metadata", {}) or {}).get("page_label", i))

        chunks.append(ParsedChunk(id=f"p{page}-text", page=page, text=md, kind="text"))

        image_paths = list(_extract_markdown_images(md, pdf_path.parent))
        captions = list(_extract_figure_captions(md))
        max_len = max(len(image_paths), len(captions))

        for fig_idx in range(1, max_len + 1):
            caption = captions[fig_idx - 1] if fig_idx - 1 < len(captions) else None
            image_path = image_paths[fig_idx - 1] if fig_idx - 1 < len(image_paths) else None
            fig = ParsedFigure(id=f"p{page}-fig{fig_idx}", page=page, caption=caption, image_path=image_path)
            figures.append(fig)
            chunks.append(
                ParsedChunk(
                    id=f"{fig.id}-ctx",
                    page=page,
                    text=f"Figure caption: {caption or '(no caption extracted)'}",
                    kind="figure_context",
                    metadata={"figure_id": fig.id},
                )
            )

    return chunks, figures




def _extract_markdown_images(markdown_text: str, base_dir: Path) -> Iterable[Path]:
    # Markdown image syntax: ![alt](path)
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", markdown_text):
        raw = m.group(1).strip().split()[0]
        raw = raw.strip("<>")
        p = Path(raw)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        yield p

def _extract_figure_captions(markdown_text: str) -> Iterable[str]:
    lines = [ln.strip() for ln in markdown_text.splitlines()]
    for ln in lines:
        if re.match(r"^(figure|fig\.)\s*\d+", ln.lower()):
            yield ln


def attach_xray_findings(chunks: list[ParsedChunk], findings: list[XRayFinding]) -> list[ParsedChunk]:
    by_fig = {f.figure_id: f for f in findings}
    out: list[ParsedChunk] = []
    for c in chunks:
        out.append(c)
        fig_id = c.metadata.get("figure_id") if c.metadata else None
        if fig_id and fig_id in by_fig:
            f = by_fig[fig_id]
            out.append(
                ParsedChunk(
                    id=f"{fig_id}-finding",
                    page=f.page,
                    kind="figure_context",
                    text=f"Dental X-ray finding summary: {f.summary}\nStructured: {json.dumps(f.structured, ensure_ascii=False)}",
                    metadata={"figure_id": fig_id, "source": "xray_model"},
                )
            )
    return out


def answer_question(query: str, retriever: HybridRetriever, top_k: int = 5) -> dict[str, Any]:
    hits = retriever.search(query, k=top_k)
    context = "\n\n".join([f"[{h[0].id} | p{h[0].page}] {h[0].text[:1200]}" for h in hits])

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Please install openai package.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for answering.")

    client = OpenAI(api_key=api_key)
    prompt = (
        "Answer the question using ONLY the context below. If unknown, say unknown. "
        "Cite chunk IDs in square brackets.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt, max_output_tokens=500)

    return {
        "answer": getattr(resp, "output_text", "").strip(),
        "sources": [{"chunk_id": h[0].id, "page": h[0].page, "score": h[1]} for h in hits],
    }


def run_pipeline(pdf_path: Path, question: str) -> dict[str, Any]:
    chunks, figures = parse_pdf_with_llamaparse(pdf_path)

    xray_client = DentalXRayModelClient()
    findings: list[XRayFinding] = []

    for fig in figures:
        if fig.image_path and fig.image_path.exists() and xray_client.detect_xray(fig.image_path):
            findings.append(xray_client.analyze_xray_for_figure(fig))

    enriched = attach_xray_findings(chunks, findings)
    retriever = HybridRetriever(enriched, OpenAIEmbedder())
    return answer_question(question, retriever)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF layout parsing + dental X-ray grounded QA")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    result = run_pipeline(args.pdf, args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
