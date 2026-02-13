from __future__ import annotations

import os

import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv()

from patronus.embed import embed_batch, embed_text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
]

EXPECTED_DIM = 1536


@pytest.fixture(autouse=True)
def _reset_client() -> None:
    import patronus.embed as mod
    mod._client = None


class TestEmbedTextIntegration:
    def test_returns_correct_shape_and_dtype(self) -> None:
        result = embed_text("The quick brown fox jumps over the lazy dog.")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (EXPECTED_DIM,)

    def test_nonzero_values(self) -> None:
        result = embed_text("Machine learning and neural networks.")
        assert np.linalg.norm(result) > 0

    def test_similar_texts_have_high_similarity(self) -> None:
        a = embed_text("Mechanistic interpretability of neural networks")
        b = embed_text("Understanding how neural networks work internally")
        c = embed_text("Recipe for chocolate cake with buttercream frosting")

        sim_ab = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        sim_ac = float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)))

        assert sim_ab > sim_ac

    def test_deterministic(self) -> None:
        text = "Determinism test"
        a = embed_text(text)
        b = embed_text(text)
        np.testing.assert_array_equal(a, b)


class TestEmbedBatchIntegration:
    def test_returns_correct_count_and_shapes(self) -> None:
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        results = embed_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, np.ndarray)
            assert r.dtype == np.float32
            assert r.shape == (EXPECTED_DIM,)

    def test_batch_matches_individual(self) -> None:
        texts = ["Alpha beta gamma.", "Delta epsilon zeta."]
        batch_results = embed_batch(texts)
        individual_results = [embed_text(t) for t in texts]

        for batch_r, indiv_r in zip(batch_results, individual_results):
            np.testing.assert_array_equal(batch_r, indiv_r)

    def test_empty_batch(self) -> None:
        assert embed_batch([]) == []
