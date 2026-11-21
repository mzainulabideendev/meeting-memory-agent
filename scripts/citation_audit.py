"""Citation audit script - validates all answers and brief quotes against original transcripts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.agents.citation_agent import CitationValidator
from meeting_agent.models import Citation
from meeting_agent.storage import Database


def audit_answers():
    """Validate all answers in outputs/answers.json."""
    db = Database()
    validator = CitationValidator(db)

    answers_path = Path("outputs/answers.json")
    if not answers_path.exists():
        print("ERROR: outputs/answers.json not found")
        return

    payload = json.loads(answers_path.read_text(encoding="utf-8"))
    answers = payload.get("answers", [])
    print(f"=== Answers audit: {len(answers)} answers ===")

    total_quotes = 0
    valid_quotes = 0
    invalid_quotes = 0
    invalid_details = []

    for answer in answers:
        qid = answer.get("question_id", "?")
        for citation_dict in answer.get("citations", []):
            total_quotes += 1
            citation = Citation(**citation_dict)
            check = validator.validate(citation)
            if check.valid:
                valid_quotes += 1
            else:
                invalid_quotes += 1
                invalid_details.append({
                    "question_id": qid,
                    "meeting_id": citation.meeting_id,
                    "quote": citation.quote[:80],
                    "reason": check.reason,
                })

    print(f"Total quotes: {total_quotes}")
    print(f"Valid quotes: {valid_quotes}")
    print(f"Invalid quotes: {invalid_quotes}")
    if invalid_details:
        print("INVALID DETAILS:")
        for d in invalid_details:
            print(f"  {d}")
    return valid_quotes, total_quotes, invalid_details


def audit_brief():
    """Validate every quotation in the TS3012 brief."""
    db = Database()
    validator = CitationValidator(db)

    brief_path = Path("outputs/brief.md")
    if not brief_path.exists():
        print("ERROR: outputs/brief.md not found")
        return

    content = brief_path.read_text(encoding="utf-8")
    print(f"\n=== Brief audit ===")

    # Extract quotes from brief - format: > “quote” — TS3012x
    pattern = re.compile(r'>\s*\u201c([^“”\n]+)\u201d\s*\u2014\s*(TS3012[abcd])')
    matches = pattern.findall(content)

    print(f"Quotes found in brief: {len(matches)}")

    total = 0
    valid = 0
    invalid = 0
    invalid_details = []
    for quote, meeting_id in matches:
        total += 1
        citation = Citation(meeting_id=meeting_id, quote=quote.strip())
        check = validator.validate(citation)
        if check.valid:
            valid += 1
        else:
            invalid += 1
            invalid_details.append({
                "meeting_id": meeting_id,
                "quote": quote[:60],
                "reason": check.reason,
            })

    print(f"Total quotes: {total}")
    print(f"Valid quotes: {valid}")
    print(f"Invalid quotes: {invalid}")
    if invalid_details:
        for d in invalid_details:
            print(f"  INVALID: {d}")
    return valid, total, invalid


if __name__ == "__main__":
    print("=" * 60)
    print("CITATION AUDIT")
    print("=" * 60)
    a_valid, a_total, a_invalid = audit_answers()
    b_valid, b_total, b_invalid = audit_brief()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Answers: {a_valid}/{a_total} valid ({a_invalid} invalid)")
    print(f"Brief: {b_valid}/{b_total} valid ({b_invalid} invalid)")
    print(f"TOTAL: {a_valid + b_valid}/{a_total + b_total} valid")