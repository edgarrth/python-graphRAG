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
                "retrieval_strategy": "exact_reason_code_anchor+hybrid",
                "explicit_reason_codes": ["91"],
            }
        )
    )
    assert rows["Retriever"] == "hybrid"
    assert rows["Ranker híbrido"] == "linear"
    assert rows["Peso vectorial"] == "0.35"
    assert rows["Search ratio"] == "3"
    assert rows["Estrategia"] == "exact_reason_code_anchor+hybrid"
    assert rows["Códigos anclados"] == "91"
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


def test_chat_has_one_css_owned_scroll_surface_and_js_stream_follow() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert 'key="chat_scroll_panel"' in app_source
    assert '.st-key-chat_scroll_panel {' in app_source
    assert 'overflow-y:auto !important;' in app_source
    assert 'touch-action:pan-y;' in app_source
    assert 'scroll-behavior:auto !important;' in app_source
    # The viewport is the only sized element (explicit height, no dependency on
    # Streamlit wrappers), scrolls top-to-bottom, and everything between it and
    # the thread is forced to be content-sized so overflow stays scrollable.
    assert 'height:var(--axiz-chat-h, calc(100dvh - {CHAT_RESERVED_PX}px)) !important;' in app_source
    assert 'column-reverse' not in app_source
    assert '.st-key-chat_scroll_panel *:has(.st-key-chat_thread),' in app_source
    assert 'key="chat_thread"' in app_source
    assert 'chat-bottom-anchor' in app_source
    # The follow driver releases on upward scroll and is re-armed per question.
    assert 'if (pane.scrollTop < lastTop - 2) stick = false;' in app_source
    assert 'unsafe_allow_javascript' in app_source
    assert 'st.session_state.scroll_nonce += 1' in app_source
    assert 'autoscroll=True' not in app_source
    assert 'components.v1' not in app_source


def test_first_stream_hides_stale_empty_state_immediately() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert 'key="empty_state"' in app_source
    assert 'class="stream-active-marker"' in app_source
    assert '.st-key-center_shell:has(.stream-active-marker) .st-key-empty_state' in app_source
    # The welcome state is rendered only when there are no messages and no
    # pending request; once streaming starts, the marker also hides any stale
    # DOM from the previous Streamlit run until it is pruned.
    assert "elif not pending_request:" in app_source


def test_example_questions_have_dedicated_spacing_container() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert 'key="example_questions"' in app_source
    assert ".st-key-example_questions" in app_source
    assert "line-height:1.25" in app_source


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



def test_sse_rendering_batches_small_deltas_for_ui_performance() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert "STREAM_RENDER_BATCH_CHARS = 64" in app_source
    assert "pending_parts: list[str] = []" in app_source
    assert "pending_chars >= STREAM_RENDER_BATCH_CHARS" in app_source
    assert 'yield "".join(pending_parts)' in app_source

def test_chat_composer_is_a_visible_flex_row_inside_center_shell() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")

    assert 'key="center_shell"' in app_source
    assert '.st-key-center_shell {' in app_source
    assert 'div[data-testid="stElementContainer"]:has(.st-key-chat_composer)' in app_source
    assert 'flex:0 0 auto !important;' in app_source
    assert 'position:absolute !important;' not in app_source.split('/* Inline composer', 1)[1].split('/* Narrow windows', 1)[0]
    assert 'padding:.18rem 0 .22rem !important;' in app_source
    assert 'max-height:128px' in app_source

