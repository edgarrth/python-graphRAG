import ast
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
    rows = dict(
        trace_rows(
            {
                "retriever": "hybrid",
                "ranker": "linear",
                "vector_weight": 0.35,
                "effective_search_ratio": 3,
                "returned_contexts": 4,
            }
        )
    )
    assert rows["Retriever"] == "hybrid"
    assert rows["Ranker híbrido"] == "linear"
    assert rows["Peso vectorial"] == "0.35"
    assert rows["Search ratio"] == "3"
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


def test_frontend_uses_custom_collapsible_navigation_and_right_settings() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")
    config_source = (frontend / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert "left_sidebar_collapsed" in app_source
    assert 'key="collapse-left"' in app_source
    assert 'key="open-left"' in app_source
    assert 'key="left_nav_panel"' in app_source
    assert 'key="right_settings_panel"' in app_source
    assert "CHAT_MAX_WIDTH = 1040 if LEFT_COLLAPSED else 940" in app_source
    assert '[data-testid="stSidebar"]' in app_source
    assert "display:none !important" in app_source
    assert 'toolbarMode = "minimal"' in config_source


def test_right_settings_exposes_graph_rag_controls() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert "Top K de recuperación" in app_source
    assert "Actividad técnica" in app_source
    assert "Evidencia recuperada" in app_source
    assert "Progreso de consulta" in app_source
    assert "Limpiar conversación" in app_source


def test_service_status_and_generation_provider_are_rendered_left() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert "def render_left_navigation(service_ready: bool, readiness: dict[str, Any])" in app_source
    assert "left-status-card" in app_source
    assert 'status_text = "API y Neo4j disponibles" if service_ready' in app_source
    assert "Generación activa:" in app_source
    assert 'readiness.get("generation_provider"' in app_source


def test_chat_has_own_scroll_surface_and_native_autoscroll() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert 'key="chat_scroll_panel"' in app_source
    assert "height=620" in app_source
    assert "autoscroll=True" in app_source
    assert ".st-key-chat_scroll_panel" in app_source
    assert "overflow-y:auto !important" in app_source
    assert "components.v1" not in app_source
    assert "components.html" not in app_source


def test_frontend_uses_sse_stream_endpoint() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    client_source = (frontend / "api_client.py").read_text(encoding="utf-8")
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert "/api/v1/graphrag/query/stream" in client_source
    assert '"Accept": "text/event-stream"' in client_source
    assert "client.query_stream(" in app_source
    assert 'event == "delta"' in app_source


def test_frontend_app_source_is_valid_python() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")
    ast.parse(app_source)
