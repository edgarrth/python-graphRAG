"""Regression tests for the experiment example and its chat-only dispatch."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "frontend"))
from benchmark_chat import (  # noqa: E402
    BENCHMARK_EXAMPLE_QUESTION, additional_payment_rows, benchmark_top_k, query_rows, strategy_rows,
)


@pytest.mark.parametrize(("question", "expected"), [
    (BENCHMARK_EXAMPLE_QUESTION, 5),
    (" /evaluar graphsage ", 5),
    ("/EVALUAR  GraphSAGE k=10", 10),
    ("/evaluar graphsage k=1", 1),
    ("/evaluar graphsage k=20", 20),
    ("/evaluar graphsage k=21", None),
    ("¿Qué significa GraphSAGE?", None),
    ("¿GraphSAGE compara pagos?", None),
    ("/evaluar graphsage y explícame los pagos", None),
])
def test_only_explicit_experiment_question_triggers_benchmark(question: str, expected: int | None) -> None:
    assert benchmark_top_k(question) == expected


def test_additional_payments_are_shown_with_their_query_context() -> None:
    assert additional_payment_rows({"results": [
        {"anchor": "BENCH-001", "additional_relevant_ids": ["BENCH-005", "BENCH-009"]},
        {"anchor": "BENCH-002", "additional_relevant_ids": []},
    ]}) == [
        {"Consulta": "BENCH-001", "Pago adicional": "BENCH-005"},
        {"Consulta": "BENCH-001", "Pago adicional": "BENCH-009"},
    ]


def test_full_width_tables_have_short_column_names() -> None:
    metric = {"precision_at_k": 0.2, "recall_at_k": 0.5, "ndcg_at_k": 0.4}
    rows = strategy_rows({"top_k": 5, "baseline": metric, "graphsage": metric, "union_at_2k": metric})
    assert len(rows) == 3 and set(rows[0]) == {"Estrategia", "Precisión", "Recall", "nDCG"}
    assert rows[-1]["Estrategia"] == "Unión @≤10*"
    query = {"baseline": [{"payment_id": "BENCH-001", "score": 12.0, "relevant_in_synthetic_truth": True}],
             "graphsage": [{"payment_id": "BENCH-005", "score": 0.94, "relevant_in_synthetic_truth": False}]}
    values = query_rows(query)
    assert values == [
        {"Método": "Reglas", "Pago": "BENCH-001", "Puntaje": "12.000", "Relevante": "Sí"},
        {"Método": "GraphSAGE", "Pago": "BENCH-005", "Puntaje": "0.940", "Relevante": "No"},
    ]


class State(dict):
    def __getattr__(self, key: str):
        return self[key]

    def __setattr__(self, key: str, value):
        self[key] = value


def app_function(name: str, env: dict):
    """Load an unmodified function from the Streamlit script without running its UI."""
    source = (ROOT / "frontend/app.py").read_text(encoding="utf-8")
    funcs = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(funcs) == 1
    node = funcs[0]
    # Function signatures and local annotations in this file use only builtins
    # or delayed types. Compile with postponed annotation evaluation.
    ast.fix_missing_locations(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    scope = dict(env)
    exec(compile(ast.fix_missing_locations(module), "frontend/app.py", "exec"), scope)
    return scope[name]


def test_example_bypasses_llm_and_is_persisted_as_chat_turn() -> None:
    state = State(pending_request=None, scroll_nonce=0, retrieval_mode="Neural GraphRAG inteligente")
    conversation = {"id": "c1", "last_payment_id": "PAY-1008"}
    messages: list[tuple[str, str]] = []
    plan_calls: list[str] = []

    def never_plan(*args):
        plan_calls.append("called")
        raise AssertionError("Benchmark must not call chat planning or SSE")

    start = app_function("start_queued_turn", {
        "st": SimpleNamespace(session_state=state), "current_conversation": lambda: conversation,
        "benchmark_top_k": benchmark_top_k, "add_message": lambda r, c: messages.append((r, c)),
        "plan_turn": never_plan, "LOGGER": SimpleNamespace(info=lambda *args: None),
    })
    start(BENCHMARK_EXAMPLE_QUESTION)
    assert messages == [("user", BENCHMARK_EXAMPLE_QUESTION)]
    assert state.pending_turn_kind == "benchmark" and state.pending_benchmark_k == 5
    assert state.pending_conversation_id == "c1" and state.scroll_nonce == 1
    assert conversation["last_payment_id"] == "PAY-1008"  # Does not pollute payment reference.
    assert plan_calls == []
    start(BENCHMARK_EXAMPLE_QUESTION)
    assert len(messages) == 1  # A pending turn cannot be submitted twice.


@contextmanager
def fake_box(*args, **kwargs):
    yield SimpleNamespace(update=lambda **kwargs: None)


def test_report_is_stored_in_chat_and_not_requested_again_on_rerender() -> None:
    result = {"top_k": 5, "cohort_payments": 48, "queries": 8, "results": []}
    state = State(pending_request=BENCHMARK_EXAMPLE_QUESTION, pending_turn_kind="benchmark",
                  pending_benchmark_k=5, pending_conversation_id="c1", pending_conversation_payment_id=None)
    calls: list = []
    answers: list = []
    fake_st = SimpleNamespace(
        session_state=state, html=lambda html: None, chat_message=fake_box, status=fake_box,
        markdown=lambda content: calls.append(("markdown", content)),
        error=lambda msg: calls.append(("error", msg)), rerun=lambda: calls.append(("rerun",)),
    )
    client = SimpleNamespace(graphsage_value_benchmark=lambda k: (calls.append(("endpoint", k)), result)[1])
    render = app_function("render_benchmark_assistant", {
        "st": fake_st, "APP_ICON": "icon", "ApiClient": object, "httpx": httpx,
        "uuid4": lambda: UUID("00000000-0000-0000-0000-000000000001"),
        "LOGGER": SimpleNamespace(exception=lambda *a: None),
        "render_benchmark_report": lambda report, rid: calls.append(("report", report, rid)),
        "add_message": lambda role, content, payload=None: answers.append((role, content, payload)),
    })
    render(client, 5)
    assert calls[0] == ("endpoint", 5)
    assert calls[2] == ("report", result, "00000000000000000000000000000001")
    assert answers[0][2]["kind"] == "benchmark" and answers[0][2]["report"] == result
    assert state.pending_request is None and state.pending_benchmark_k is None
    assert state.pending_turn_kind == "chat" and calls[-1] == ("rerun",)


def test_benchmark_http_409_reports_missing_prerequisites_without_stale_metrics() -> None:
    request = httpx.Request("GET", "http://api:8000/bench")
    response = httpx.Response(409, request=request, json={"detail": "Entrena GraphSAGE y carga BENCH-"})
    state = State(pending_request="test", pending_turn_kind="benchmark", pending_benchmark_k=5,
                  pending_conversation_id="c1", pending_conversation_payment_id=None)
    calls: list = []
    answers: list = []
    fake_st = SimpleNamespace(session_state=state, html=lambda html: None, chat_message=fake_box,
                              status=fake_box, error=lambda s: calls.append(s), rerun=lambda: None)
    render = app_function("render_benchmark_assistant", {
        "st": fake_st, "APP_ICON": "icon", "httpx": httpx,
        "LOGGER": SimpleNamespace(exception=lambda *a: None),
        "render_benchmark_report": lambda *a: pytest.fail("No report after 409"),
        "add_message": lambda *args: answers.append(args),
    })
    def failure(k):
        raise httpx.HTTPStatusError("not ready", request=request, response=response)
    render(SimpleNamespace(graphsage_value_benchmark=failure), 5)
    assert len(calls) == 1 and "Entrena GraphSAGE y carga BENCH-" in calls[0]
    assert answers[0][0] == "assistant" and len(answers[0]) == 2
    assert state.pending_request is None


def test_benchmark_ui_is_outside_sidebar_and_full_width() -> None:
    source = (ROOT / "frontend/app.py").read_text(encoding="utf-8")
    sidebar = source.split("def render_left_navigation(", 1)[1].split("def render_right_settings(", 1)[0]
    assert 'Prueba · GraphSAGE vs. reglas' not in sidebar
    assert 'key="benchmark-run"' not in sidebar and 'key="benchmark-k"' not in sidebar
    main = source.split("def render_chat_area(", 1)[1].split("def start_queued_turn(", 1)[0]
    assert 'if st.session_state.pending_turn_kind == "benchmark":' in main
    assert 'render_benchmark_assistant(' in main
    report = source.split("def render_benchmark_report(", 1)[1].split("def render_benchmark_assistant(", 1)[0]
    assert 'st.dataframe(query_rows(item), hide_index=True, width="stretch")' in report
    assert 'left, right = st.columns(2)' not in report
