from __future__ import annotations

import os
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

_client: Optional[OpenAI] = None
_DEFAULT_MODEL = "text-embedding-3-small"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_text(text: str, *, model: str = _DEFAULT_MODEL) -> np.ndarray:
    client = _get_client()
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


_MAX_CHARS_PER_BATCH = 800_000


def embed_batch(texts: list[str], *, model: str = _DEFAULT_MODEL) -> list[np.ndarray]:
    if not texts:
        return []
    client = _get_client()
    all_embeddings: list[np.ndarray] = []
    chunk: list[str] = []
    chunk_chars = 0

    for text in texts:
        if chunk and chunk_chars + len(text) > _MAX_CHARS_PER_BATCH:
            response = client.embeddings.create(input=chunk, model=model)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend(
                np.array(d.embedding, dtype=np.float32) for d in sorted_data
            )
            chunk = []
            chunk_chars = 0
        chunk.append(text)
        chunk_chars += len(text)

    if chunk:
        response = client.embeddings.create(input=chunk, model=model)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend(
            np.array(d.embedding, dtype=np.float32) for d in sorted_data
        )

    return all_embeddings
