"""Unit tests for EmpirioLabsEmbeddings. These run without network access."""

import os
from typing import Any

from langchain_empiriolabs import EmpirioLabsEmbeddings
from langchain_empiriolabs.embeddings import DEFAULT_API_BASE

os.environ.setdefault("EMPIRIOLABS_API_KEY", "test-key")


class _RecordingClient:
    """Stands in for the OpenAI embeddings client and records each request's inputs."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def create(self, input: list[str], **kwargs: Any) -> dict:  # noqa: A002
        self.batches.append(list(input))
        batch = float(len(self.batches))
        return {"data": [{"embedding": [batch, float(i)]} for i in range(len(input))]}


def test_defaults() -> None:
    embeddings = EmpirioLabsEmbeddings()
    assert embeddings.model == "text-embedding-v4"
    assert embeddings.openai_api_base == DEFAULT_API_BASE
    # LangChain's own default; the API takes up to 2,048 inputs per request.
    assert embeddings.chunk_size == 1000


def test_embed_documents_sends_one_request_per_chunk_in_order() -> None:
    embeddings = EmpirioLabsEmbeddings()
    client = _RecordingClient()
    embeddings.client = client
    texts = [f"document {i}" for i in range(1500)]

    vectors = embeddings.embed_documents(texts)

    assert [len(batch) for batch in client.batches] == [1000, 500]
    assert [text for batch in client.batches for text in batch] == texts
    assert len(vectors) == 1500


def test_chunk_size_can_be_set() -> None:
    embeddings = EmpirioLabsEmbeddings(chunk_size=16)
    client = _RecordingClient()
    embeddings.client = client

    embeddings.embed_documents([f"text {i}" for i in range(40)])

    assert [len(batch) for batch in client.batches] == [16, 16, 8]
