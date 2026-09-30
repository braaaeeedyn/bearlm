"""RAGAS evaluation — automated, reference-free answer grading (fully local).

RAGAS (Retrieval-Augmented Generation Assessment) uses an LLM as a judge to score
a RAG system's answers. We run it entirely on the local Ollama model, so no
OpenAI/keys are needed. Two reference-free metrics (no human-written gold answer
required) replace the manual grading in answer_eval.py:

  - faithfulness      : is every claim in the answer supported by the retrieved
                        context?  (0..1)  -> catches hallucination.
  - answer_relevancy  : does the answer actually address the question?  (0..1)

For each eval question we run the real local pipeline (retrieve -> grounded
answer), then hand RAGAS {question, answer, retrieved_contexts}.

Run (small sample; each RAGAS sample = several judge LLM calls, so it is slow):
    BACKEND=local uv run python eval/ragas_eval.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# --- shim: ragas 0.3.1 imports a module the newer langchain-community moved.
#     We never use Vertex AI (judge is local Ollama); a stub lets ragas import.
_vx = types.ModuleType("langchain_community.chat_models.vertexai")
_vx.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vx)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

from bearlm import config  # noqa: E402


def _answer_with_contexts(question: str):
    """Run the real local corpus pipeline; return (answer, [context strings])."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from bearlm import rag

    docs = rag._retrieve_local(question)
    contexts = [d.page_content for d in docs]
    if not docs:
        return "I don't know based on the course materials.", contexts
    prompt = ChatPromptTemplate.from_messages(
        [("system", rag._STRICT_SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")]
    )
    answer = (prompt | rag._get_llm() | StrOutputParser()).invoke(
        {"context": rag._format_docs(docs), "question": question}
    )
    return answer, contexts


def main() -> None:
    ap = argparse.ArgumentParser()
    default_q = _common.EVAL_DIR / "questions_generated.jsonl"
    ap.add_argument("--questions", default=str(default_q if default_q.exists()
                                               else _common.EVAL_DIR / "questions.jsonl"))
    ap.add_argument("--limit", type=int, default=8, help="number of questions (RAGAS is slow)")
    ap.add_argument("--out", default=str(_common.EVAL_DIR / "out" / "ragas.json"))
    args = ap.parse_args()

    if config.BACKEND != "local":
        sys.exit("Run with BACKEND=local.")

    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import RunConfig, evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy

    questions = _common.load_questions(args.questions)[: args.limit]
    print(f"Running the local pipeline on {len(questions)} questions…")
    samples = []
    for i, q in enumerate(questions, 1):
        answer, contexts = _answer_with_contexts(q["question"])
        samples.append(SingleTurnSample(
            user_input=q["question"], response=answer, retrieved_contexts=contexts,
        ))
        print(f"  [{i}/{len(questions)}] {q['question'][:60]}")

    dataset = EvaluationDataset(samples=samples)
    judge = LangchainLLMWrapper(ChatOllama(
        model=config.OLLAMA_CHAT_MODEL, temperature=0,
        num_ctx=config.OLLAMA_NUM_CTX, base_url=config.OLLAMA_BASE_URL,
    ))
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL,
    ))

    print("\nScoring with RAGAS (faithfulness, answer_relevancy)…")
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=judge, embeddings=emb,
        run_config=RunConfig(max_workers=2, timeout=240),
        show_progress=True,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c in ("faithfulness", "answer_relevancy")]
    means = {c: float(df[c].mean(skipna=True)) for c in metric_cols}

    print("\n" + "=" * 60)
    print("RAGAS results (local judge, n=%d)" % len(questions))
    print("=" * 60)
    for c in metric_cols:
        scored = int(df[c].notna().sum())
        print(f"  {c:18} {means[c]:.3f}   (scored {scored}/{len(df)})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n": len(questions), "means": means,
        "per_question": df[["user_input", *metric_cols]].to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
