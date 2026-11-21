"""Command-line interface for the meeting memory agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from meeting_agent.config import get_settings
from meeting_agent.llm.provider import provider_status
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_ingest(args: argparse.Namespace) -> int:
    from meeting_agent.workflows.ingestion_workflow import IngestionWorkflow

    workflow = IngestionWorkflow()
    stats, report = workflow.run(
        reset=args.reset,
        limit=args.limit,
        extract=not args.no_extract,
        transcripts_dir=Path(args.data) if args.data else None,
    )
    _print(
        {
            "files_ingested": stats.files_ingested,
            "files_skipped": stats.files_skipped,
            "files_failed": stats.files_failed,
            "meetings": stats.meetings,
            "chunks": stats.chunks,
            "words": stats.words,
            "embedding_seconds": round(stats.embedding_seconds, 2),
            "total_seconds": round(stats.total_seconds, 2),
            "total_meetings_in_db": report["database_meetings"],
            "total_chunks_in_db": report["database_chunks"],
            "report": str(get_settings().outputs_dir / "dataset_report.json"),
        }
    )
    return 0 if stats.files_failed == 0 else 1


def cmd_answer(args: argparse.Namespace) -> int:
    from meeting_agent.workflows.answer_workflow import AnswerWorkflow

    settings = get_settings()
    questions_path = Path(args.questions) if args.questions else settings.resolved_questions_path
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    if args.limit:
        questions = questions[: args.limit]

    workflow = AnswerWorkflow(settings)
    results = asyncio.run(workflow.run_questions(questions, concurrency=args.concurrency))

    out_path = Path(args.out) if args.out else settings.outputs_dir / "answers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    brief_text = ""
    brief_path = settings.outputs_dir / "brief.md"
    if brief_path.exists():
        brief_text = brief_path.read_text(encoding="utf-8")

    output = {
        "answers": [
            {
                "question_id": result.question_id,
                "answer": result.answer,
                "citations": [c.model_dump() for c in result.citations],
            }
            for result in results
        ],
        "brief": brief_text,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    _print(
        {
            "questions": len(results),
            "answered_with_citations": sum(1 for r in results if r.citations),
            "insufficient_evidence": sum(1 for r in results if r.insufficient_evidence),
            "output": str(out_path),
            "provider": workflow.llm.provider_name,
            "llm_calls": workflow.llm.calls,
            "input_tokens": workflow.llm.input_tokens,
            "output_tokens": workflow.llm.output_tokens,
        }
    )
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    from meeting_agent.workflows.briefing_workflow import BriefingWorkflow

    settings = get_settings()
    result = BriefingWorkflow(settings).run_sync(args.team)
    print(result.markdown)
    _print(
        {
            "team": result.team,
            "meetings_used": result.meetings_used,
            "citations": len(result.citations),
            "output": str(settings.outputs_dir / "brief.md"),
        }
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from meeting_agent.evaluation.evaluator import run_evaluation

    report = asyncio.run(run_evaluation(regenerate=not args.use_saved))
    print(report["summary_markdown"])
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from meeting_agent.workflows.answer_workflow import AnswerWorkflow

    workflow = AnswerWorkflow()
    result = workflow.run_sync("LIVE", args.question, args.team)
    _print(
        {
            "answer": result.answer,
            "citations": [c.model_dump() for c in result.citations],
            "meetings": result.evidence_meetings,
            "insufficient_evidence": result.insufficient_evidence,
            "trace": result.retrieval_trace,
        }
    )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from meeting_agent.ingestion.pipeline import build_dataset_report
    from meeting_agent.storage import Database

    report = build_dataset_report(database=Database())
    path = get_settings().outputs_dir / "dataset_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(
        {
            "total_transcript_files": report["total_transcript_files"],
            "total_teams": report["total_teams"],
            "total_meetings": report["total_meetings"],
            "total_words_body_only": report["total_words_body_only"],
            "teams_with_missing_meetings": report["teams_with_missing_meetings"],
            "malformed": len(report["malformed_transcripts"]),
            "questions_count": report["questions_count"],
            "output": str(path),
        }
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from meeting_agent.storage import Database
    from meeting_agent.memory.vector_store import VectorStore

    db = Database()
    try:
        vectors = VectorStore().count()
    except Exception as exc:
        vectors = f"unavailable: {exc}"
    _print(
        {
            "provider": provider_status(),
            "meetings": db.meeting_count(),
            "chunks": db.chunk_count(),
            "vectors": vectors,
            "teams": db.list_teams(),
            "citation_attempts": db.citation_stats(),
            "metrics": db.get_metrics(),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meeting_agent", description="Agentic AI meeting memory")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Ingest and index transcripts")
    p.add_argument("--data", help="Transcript directory")
    p.add_argument("--reset", action="store_true", help="Clear existing memory before ingest")
    p.add_argument("--limit", type=int, help="Limit number of files (testing)")
    p.add_argument("--no-extract", action="store_true", help="Skip semantic extraction")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("answer", help="Answer assessment questions")
    p.add_argument("--questions", help="questions.json path")
    p.add_argument("--out", help="output answers.json path")
    p.add_argument("--limit", type=int, help="Limit number of questions")
    p.add_argument("--concurrency", type=int, default=1)
    p.set_defaults(func=cmd_answer)

    p = sub.add_parser("brief", help="Generate a team briefing")
    p.add_argument("--team", default="TS3012")
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser("evaluate", help="Run independent evaluation")
    p.add_argument("--use-saved", action="store_true", help="Evaluate saved answers instead of re-running")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("ask", help="Ask a single ad-hoc question")
    p.add_argument("question")
    p.add_argument("--team")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("inspect", help="Inspect the dataset and write dataset_report.json")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("status", help="Show memory and provider status")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
