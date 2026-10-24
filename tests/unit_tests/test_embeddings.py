"""Unit tests for EmpirioLabsEmbeddings. These run without network access."""

import os
from typing import Any

from langchain_empiriolabs import EmpirioLabsEmbeddings
from langchain_empiriolabs.embeddings import DEFAULT_API_BASE, DEFAULT_CHUNK_SIZE

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
    assert DEFAULT_CHUNK_SIZE == 10
    assert embeddings.chunk_size == DEFAULT_CHUNK_SIZE


def test_embed_documents_sends_at_most_ten_inputs_per_request() -> None:
    embeddings = EmpirioLabsEmbeddings()
    client = _RecordingClient()
    embeddings.client = client
    texts = [f"document {i}" for i in range(25)]

    vectors = embeddings.embed_documents(texts)

    assert [len(batch) for batch in client.batches] == [10, 10, 5]
    assert [text for batch in client.batches for text in batch] == texts
    assert len(vectors) == 25


def test_chunk_size_can_be_raised_for_a_model_that_takes_more() -> None:
    embeddings = EmpirioLabsEmbeddings(model="skylark-embedding-vision", chunk_size=16)
    client = _RecordingClient()
    embeddings.client = client

    embeddings.embed_documents([f"text {i}" for i in range(16)])

    assert [len(batch) for batch in client.batches] == [16]
