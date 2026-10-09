import pytest
from rank_bm25 import BM25Okapi

from bearlm import search


@pytest.mark.parametrize("text,tokens", [
    ("B+ tree", ["b+", "tree"]),
    ("B+-tree", ["b+", "tree"]),
    ("C++", ["c++"]),
    ("C#", ["c#"]),
    ("x+1", ["x", "1"]),
    ("a+b", ["a", "b"]),
    ("#include", ["include"]),
])
def test_tokenizer_keeps_symbol_terms(text, tokens):
    assert search._tokenize(text) == tokens


def test_bm25_ranks_symbol_term_doc():
    texts = [
        "A binary tree has at most two children per node, and a binary search tree orders its keys.",
        "A B+ tree is a balanced search tree that keeps every record in its leaf pages.",
        "Grade b means good work; a b grade is above c and below a.",
    ]
    bm25 = BM25Okapi([search._tokenize(t) for t in texts])
    scores = bm25.get_scores(search._tokenize("What is a B+ tree?"))
    assert max(range(len(texts)), key=lambda i: scores[i]) == 1
