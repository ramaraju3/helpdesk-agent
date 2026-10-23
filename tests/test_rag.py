"""Tests for the RAG pipeline in mcp_server/rag.py.

Chunking tests need no model. The retrieval tests use the real local embedding
model, which is free and deterministic, so they double as a tiny retrieval eval.
"""

import pytest

from mcp_server.rag import chunk_markdown, load_chunks, search
from mcp_server.server import search_docs

# A small labeled eval set: a question in everyday words -> the doc that answers it.
# The queries avoid the docs' own wording, so keyword search would miss most of them.
RETRIEVAL_EVAL = {
    "my computer is completely dead and won't boot": "laptop-power-and-battery.md",
    "battery is puffing up": "laptop-power-and-battery.md",
    "can I email a customer spreadsheet to my gmail": "data-handling-policy.md",
    "copy files to a thumb drive": "data-handling-policy.md",
    "going on a work trip to Japan next month": "vpn-and-remote-access.md",
    "first day at the company, login says inactive": "new-hire-onboarding.md",
    "people on the video call can't hear us": "conference-room-av.md",
    "someone is quitting, what happens to their files": "offboarding.md",
    "is it ok to use an MIT licensed library in our product": "software-licensing.md",
    "who gets paged for a major incident": "incident-response.md",
}


def test_chunk_markdown_splits_on_sections():
    doc = "# Title\n\nIntro is ignored.\n\n## First\nline one\nline two\n\n## Second\nbody"
    chunks = chunk_markdown("doc.md", doc)
    assert [(c.section, c.text) for c in chunks] == [
        ("First", "line one line two"),
        ("Second", "body"),
    ]
    assert chunks[0].title == "Title"
    assert chunks[0].for_embedding() == "Title - First\nline one line two"


def test_all_docs_are_chunked():
    chunks = load_chunks()
    assert len({c.source for c in chunks}) == 8
    assert all(c.title and c.section and c.text for c in chunks)


@pytest.mark.parametrize("query,expected_source", RETRIEVAL_EVAL.items())
def test_retrieval_hit_at_3(query, expected_source):
    results = search(query, k=3)
    assert expected_source in [r["source"] for r in results]


def test_results_are_sorted_by_score():
    scores = [r["score"] for r in search("vpn is slow", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_search_docs_tool_returns_sources():
    results = search_docs("laptop battery is swollen")
    assert len(results) == 3
    assert results[0]["section"] == "Swollen or hot battery"
    assert set(results[0]) == {"source", "section", "text", "score"}
