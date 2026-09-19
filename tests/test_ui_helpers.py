from datetime import UTC, datetime, timedelta

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend"))

from ui_helpers import (  # noqa: E402
    EXAMPLE_QUESTIONS,
    conversation_group,
    conversation_title,
    trace_rows,
)


def test_conversation_title_compacts_long_question() -> None:
    value = conversation_title("  ¿Por qué   fallan los pagos internacionales cuando el emisor tarda?  ", 30)
    assert value == "¿Por qué fallan los pagos int…"


def test_conversation_group_today_and_recent() -> None:
    now = datetime(2026, 9, 19, 10, 0, tzinfo=UTC)
    assert conversation_group(now - timedelta(hours=2), now) == "Hoy"
    assert conversation_group(now - timedelta(days=3), now) == "Últimos 7 días"


def test_trace_rows_has_expected_graphrag_fields() -> None:
    rows = dict(trace_rows({"retriever": "hybrid", "returned_contexts": 4}))
    assert rows["Retriever"] == "hybrid"
    assert rows["Contextos retornados"] == "4"


def test_example_questions_are_kept_for_empty_state() -> None:
    assert len(EXAMPLE_QUESTIONS) == 4
    assert any("código 05" in question for question in EXAMPLE_QUESTIONS)
    assert any("idempotencia" in question for question in EXAMPLE_QUESTIONS)


def test_axiz_reference_assets_are_packaged() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    assert (frontend / "assets" / "axiz-agent-icon.png").is_file()
    assert (frontend / "assets" / "axiz-logo@2x.png").is_file()
    assert (frontend / "assets" / "favicon.png").is_file()
    assert (frontend / ".streamlit" / "config.toml").is_file()
