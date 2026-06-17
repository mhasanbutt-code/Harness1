"""Load and chunk project documents for RAG."""

from __future__ import annotations

from pathlib import Path

from harness.rag.store import Chunk

TEXT_EXTS = {".md", ".txt", ".rst", ".py", ".json", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", "outputs", "node_modules"}


def load_documents(paths: list[str | Path]) -> list[tuple[str, str]]:
    """Return ``(source, text)`` for every text file under the given paths."""
    docs: list[tuple[str, str]] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if (
                    f.is_file()
                    and f.suffix.lower() in TEXT_EXTS
                    and not (set(f.parts) & SKIP_PARTS)
                ):
                    docs.append((str(f), _read(f)))
        elif p.is_file():
            docs.append((str(p), _read(p)))
    return [(s, t) for s, t in docs if t.strip()]


def _read(f: Path) -> str:
    try:
        return f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks, preferring newline boundaries."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            nl = text.rfind("\n", start + int(size * 0.6), end)
            if nl != -1:
                end = nl
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def make_chunks(paths: list[str | Path], size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Load, chunk, and label every document under the given paths."""
    out: list[Chunk] = []
    for source, text in load_documents(paths):
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            out.append(Chunk(id=f"{source}#{i}", text=piece, source=source))
    return out
