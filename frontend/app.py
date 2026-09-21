from __future__ import annotations

import inspect
import json
from collections import OrderedDict
from datetime import UTC, datetime
from html import escape
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st
from api_client import ApiClient
from benchmark_chat import (
    BENCHMARK_EXAMPLE_QUESTION, additional_payment_rows, benchmark_top_k,
    query_rows, strategy_rows,
)
from chat_flow import plan_turn
from ui_helpers import EXAMPLE_QUESTIONS, conversation_group, conversation_title, graphsage_usage, trace_rows

LOGGER = logging.getLogger("axiz.graphrag.frontend")
logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
APP_ICON_PATH = ASSET_DIR / "axiz-agent-icon.png"
AXIZ_LOGO_PATH = ASSET_DIR / "axiz-logo@2x.png"
FAVICON_PATH = ASSET_DIR / "favicon.png"

APP_ICON = str(APP_ICON_PATH)
AXIZ_LOGO = str(AXIZ_LOGO_PATH)
FAVICON = str(FAVICON_PATH)

st.set_page_config(
    page_title="Axiz GraphRAG Payments",
    page_icon=FAVICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


for key, default in {
    "conversations": {},
    "current_conversation_id": None,
    "pending_question": None,
    "submitted_chat_question": None,
    "pending_request": None,
    "show_trace": True,
    "show_query_progress": True,
    "show_evidence": True,
    "top_k": 4,
    "retrieval_mode": "Neural GraphRAG inteligente",
    "pending_conversation_payment_id": None,
    "pending_retrieval_mode": "traditional",
    "pending_turn_kind": "chat",
    "pending_benchmark_k": None,
    "pending_conversation_id": None,
    "left_sidebar_collapsed": False,
    "scroll_nonce": 0,
}.items():
    st.session_state.setdefault(key, default)


LEFT_COLLAPSED = bool(st.session_state.left_sidebar_collapsed)
CHAT_MAX_WIDTH = 1040 if LEFT_COLLAPSED else 940
STREAM_RENDER_BATCH_CHARS = 64
# Fallback for the vertical space (px) that is NOT the message viewport (page
# padding, top bar, composer, gaps, "☰" row). When the scroll driver JS runs it
# measures the real value and overrides it through --axiz-chat-h.
CHAT_RESERVED_PX = 284 + (56 if LEFT_COLLAPSED else 0)

st.html(
    f"""
<style>
:root {{
  color-scheme: dark;
  --axiz-bg:#081018;
  --axiz-sidebar:#09141e;
  --axiz-panel:#0b1721;
  --axiz-card:#0d1b26;
  --axiz-card-user:#171a2d;
  --axiz-line:#1d3446;
  --axiz-line-strong:#29465a;
  --axiz-text:#dce8f0;
  --axiz-muted:#7890a3;
  --axiz-accent:#43c3ec;
  --axiz-success:#4bd8a0;
}}

/* Desktop shell: the document itself never scrolls. Only the three rails may
   scroll internally, which prevents navigation/settings from disappearing. */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {{
  height:100dvh !important;
  max-height:100dvh !important;
  overflow:hidden !important;
  background:var(--axiz-bg) !important;
  color:var(--axiz-text);
}}
html, body {{ color-scheme:dark; }}
.stApp {{
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}}
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stMainMenu"],
.stDeployButton,
footer {{ display:none !important; }}

[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {{ display:none !important; }}

[data-testid="stMainBlockContainer"],
.block-container {{
  width:100% !important;
  max-width:1800px !important;
  height:100dvh !important;
  max-height:100dvh !important;
  overflow:hidden !important;
  box-sizing:border-box !important;
  margin-inline:auto !important;
  padding:.72rem 1rem .72rem !important;
}}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
.block-container > [data-testid="stVerticalBlock"] {{
  height:100% !important;
  min-height:0 !important;
}}

h1,h2,h3,h4 {{ color:#edf5fa !important; letter-spacing:-.015em; }}
p,li,label,[data-testid="stCaptionContainer"] {{ color:#b7c8d4; }}
a {{ color:var(--axiz-accent); }}

/* Independent side rails. They remain fully anchored to the viewport while
   the conversation scrolls in the middle. */
.st-key-left_nav_panel,
.st-key-right_settings_panel {{
  height:calc(100dvh - 2.35rem) !important;
  max-height:calc(100dvh - 2.35rem) !important;
  min-height:0 !important;
  overflow-y:auto !important;
  overscroll-behavior:contain;
  scrollbar-gutter:stable;
  background:linear-gradient(180deg,#0a1620 0%,#09131d 100%);
  border:1px solid var(--axiz-line);
  border-radius:16px;
  box-shadow:0 16px 42px rgba(0,0,0,.18);
  padding:.78rem .72rem .9rem;
  box-sizing:border-box;
}}
.st-key-left_nav_panel::-webkit-scrollbar,
.st-key-right_settings_panel::-webkit-scrollbar {{ width:7px; }}
.st-key-left_nav_panel::-webkit-scrollbar-thumb,
.st-key-right_settings_panel::-webkit-scrollbar-thumb {{
  background:#1c3547;
  border-radius:999px;
}}

/* The center column flows naturally: top bar, message viewport, composer.
   Nothing here depends on Streamlit's internal wrappers; the ONLY sized
   element is the message viewport itself (explicit viewport-based height). */
.st-key-center_shell {{
  height:auto !important;
  max-height:none !important;
  min-height:0 !important;
  overflow:visible !important;
  position:relative !important;
  box-sizing:border-box !important;
}}

/* One and only one scroll surface for the conversation: a regular keyed
   Streamlit container whose overflow and bottom anchoring are owned by CSS. */
.st-key-chat_scroll_panel {{
  width:100%;
  max-width:{CHAT_MAX_WIDTH}px;
  /* Plain top-to-bottom column with an explicit height: overflow always goes
     downwards, which is the direction every browser can scroll. */
  display:flex !important;
  flex-direction:column !important;
  flex-wrap:nowrap !important;
  flex:0 0 auto !important;
  height:var(--axiz-chat-h, calc(100dvh - {CHAT_RESERVED_PX}px)) !important;
  max-height:var(--axiz-chat-h, calc(100dvh - {CHAT_RESERVED_PX}px)) !important;
  min-height:220px !important;
  margin-inline:auto;
  padding:.15rem .42rem 1rem .08rem !important;
  border:0 !important;
  background:transparent !important;
  box-sizing:border-box;
  overflow-y:auto !important;
  overflow-x:hidden !important;
  overscroll-behavior-y:contain;
  scrollbar-gutter:stable;
  scroll-behavior:auto !important;
  touch-action:pan-y;
}}
/* Everything between the viewport and the thread must be content-sized and
   unclipped. Streamlit wrappers ship with height:100% / flex:1 1 0%, which
   pins them to the viewport height and lets the thread spill out of them. */
.st-key-chat_scroll_panel > *,
.st-key-chat_scroll_panel *:has(.st-key-chat_thread),
.st-key-chat_thread {{
  flex:0 0 auto !important;
  width:100% !important;
  height:auto !important;
  min-height:0 !important;
  max-height:none !important;
  overflow:visible !important;
}}
.st-key-chat_thread > * {{ flex-shrink:0 !important; }}
/* CSS-only bottom pinning (works even if the JS driver cannot run): the 1px
   anchor after the thread is the only scroll-anchoring candidate, so once the
   user is at the bottom the browser keeps it there while the answer grows. */
.st-key-chat_scroll_panel > *:not(:has(.chat-bottom-anchor)),
.st-key-chat_thread {{ overflow-anchor:none; }}
.chat-bottom-anchor {{ height:1px; overflow-anchor:auto; }}
.st-key-chat_scroll_panel::-webkit-scrollbar {{ width:9px; }}
.st-key-chat_scroll_panel::-webkit-scrollbar-track {{ background:transparent; }}
.st-key-chat_scroll_panel::-webkit-scrollbar-thumb {{
  background:#274052;
  border:2px solid transparent;
  background-clip:padding-box;
  border-radius:999px;
}}
.st-key-chat_scroll_panel::-webkit-scrollbar-thumb:hover {{ background:#365970; background-clip:padding-box; }}

/* Invisible host of the scroll driver script. It never owns wheel events. */
.st-key-stream_scroll_driver {{
  position:absolute !important;
  width:1px !important;
  height:1px !important;
  min-height:0 !important;
  overflow:hidden !important;
  opacity:0 !important;
  pointer-events:none !important;
  margin:0 !important;
  padding:0 !important;
}}
.st-key-stream_scroll_driver iframe {{
  width:1px !important;
  height:1px !important;
  border:0 !important;
  opacity:0 !important;
  pointer-events:none !important;
}}

/* Streamlit keeps stale elements from the previous run until the current run
   advances far enough. Hide the previous empty-state immediately when a real
   response starts so the welcome screen never remains underneath streaming. */
.st-key-center_shell:has(.stream-active-marker) .st-key-empty_state {{
  display:none !important;
}}

.panel-header {{ display:flex; align-items:center; justify-content:space-between; gap:.5rem; margin:.05rem 0 .55rem; }}
.panel-title {{ color:#dce8f0; font-size:.76rem; font-weight:760; letter-spacing:.055em; text-transform:uppercase; }}
.panel-subtitle {{ color:#657e91; font-size:.68rem; line-height:1.35; margin:-.2rem 0 .65rem; }}
.sidebar-brand {{ color:#f0f7fb; font-size:.78rem; font-weight:720; text-align:center; margin:.08rem 0 .72rem; }}
.session-group {{ color:#667f92 !important; font-size:.66rem; font-weight:750; letter-spacing:.08em; margin:.82rem 0 .22rem; text-transform:uppercase; }}
.session-caption {{ color:#587084 !important; font-size:.63rem; margin:-.4rem 0 .22rem .28rem; }}

.st-key-left_nav_panel .stButton button,
.st-key-right_settings_panel .stButton button {{ border:1px solid #203747; border-radius:9px; background:#0e1d29; color:#cbd9e3; }}
.st-key-left_nav_panel .stButton button {{ text-align:left; }}
.st-key-left_nav_panel .stButton button[kind="primary"] {{ background:#14384a; border-color:#2c91b0; color:#8ce5ff; }}
.st-key-left_nav_panel .stTextInput input {{ background:#07121b; border-color:#294154; color:var(--axiz-text); }}
.st-key-left_nav_panel hr,.st-key-right_settings_panel hr {{ border-color:var(--axiz-line); }}

.left-status-card {{ display:flex; align-items:center; gap:.55rem; margin:.15rem 0 .45rem; padding:.62rem .68rem; border:1px solid #214254; border-radius:11px; background:#0b1a24; color:#bfd0da; font-size:.73rem; line-height:1.35; }}
.left-status-card.ready {{ border-color:#1f5b48; background:#0c211d; }}
.left-status-card.offline {{ border-color:#60443a; background:#211714; }}
.left-status-indicator {{ width:8px; height:8px; flex:0 0 8px; border-radius:50%; background:#748797; }}
.left-status-card.ready .left-status-indicator {{ background:var(--axiz-success); box-shadow:0 0 8px rgba(75,216,160,.42); }}
.left-status-card.offline .left-status-indicator {{ background:#f0a071; }}
.left-tech-row {{ margin:0 0 .58rem; }}
.left-tech-row .tech-pill {{ font-size:.61rem; padding:.16rem .38rem; }}

.st-key-left_reopen_row {{ margin:0 0 .3rem; }}
.st-key-left_reopen_row .stButton > button {{ width:42px !important; min-width:42px !important; height:40px !important; min-height:40px !important; padding:0 !important; border:1px solid var(--axiz-line-strong) !important; border-radius:10px !important; background:#0d1d29 !important; color:#a9c2d2 !important; box-shadow:0 7px 22px rgba(0,0,0,.2) !important; }}
.st-key-left_reopen_row .stButton > button:hover {{ border-color:#3a9cbc !important; background:#123044 !important; color:#8ce5ff !important; }}

.axiz-topbar {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-end; width:100%; max-width:{CHAT_MAX_WIDTH}px; margin:0 auto .55rem; padding:.08rem 0 .7rem; border-bottom:1px solid var(--axiz-line); }}
.axiz-topbar h1 {{ margin:0; font-size:1.14rem; line-height:1.25; }}
.axiz-topbar .status {{ margin-top:.28rem; color:#728b9e; font-size:.71rem; }}
.axiz-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--axiz-success); box-shadow:0 0 9px rgba(75,216,160,.45); margin-right:7px; }}
.axiz-dot.offline {{ background:#748797; box-shadow:none; }}
.axiz-chips {{ display:flex; gap:7px; flex-wrap:wrap; justify-content:flex-end; }}
.axiz-chip {{ border:1px solid #243d4e; background:#0d1d29; color:#86a0b2; border-radius:999px; padding:.32rem .58rem; font:.66rem ui-monospace,SFMono-Regular,Consolas,monospace; }}

.hero-spacer {{ height:2.5vh; }}
.hero-title {{ color:#edf5fa; font-size:1.72rem; font-weight:780; letter-spacing:-.035em; margin:.62rem 0 .32rem; }}
.hero-subtitle {{ color:#7890a3; font-size:.92rem; line-height:1.55; max-width:650px; margin:0 auto; }}
.suggestion-label {{ display:block; position:relative; z-index:2; color:#667f92; font-size:.7rem; font-weight:750; line-height:1.25; letter-spacing:.06em; margin:1.05rem 0 .35rem; padding:0 0 .28rem; text-transform:uppercase; }}
.st-key-example_questions {{ position:relative; z-index:1; margin-top:.12rem; padding-top:.18rem; }}
.st-key-example_questions [data-testid="stButton"] {{ margin-top:.08rem; }}

[data-testid="stChatMessage"] {{ width:100%; max-width:{CHAT_MAX_WIDTH}px; margin-inline:auto; background:transparent; border:0; padding:.33rem .08rem .85rem; }}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {{ background:var(--axiz-card); border:1px solid var(--axiz-line); border-radius:13px; padding:.95rem 1.05rem; box-shadow:0 14px 34px rgba(0,0,0,.14); }}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{ background:var(--axiz-card-user); border-color:#323653; max-width:82%; margin-left:auto; }}
[data-testid="stChatMessageAvatarUser"],[data-testid="stChatMessageAvatarAssistant"] {{ border:1px solid #29445a; border-radius:9px; background:#0f2230; }}

[data-testid="stExpander"], [data-testid="stStatusWidget"] {{ background:#0a1721; border:1px solid var(--axiz-line); border-radius:11px; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--axiz-line); border-radius:10px; overflow:hidden; }}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {{ background:var(--axiz-accent); border-color:var(--axiz-accent); color:#041018; font-weight:750; }}
.stButton>button {{ border-radius:9px; border-color:#294154; background:#101e2a; color:#c6d7e3; }}
.stTextInput input,.stTextArea textarea {{ background:#091721; border-color:#294154; color:#e4eef5; }}
.stTextInput input::placeholder,.stTextArea textarea::placeholder {{ color:#60788a; opacity:1; }}
.stTextInput input:focus,.stTextArea textarea:focus {{ border-color:#3a9cbc; box-shadow:0 0 0 1px #3a9cbc; }}
[data-testid="stMain"] .stButton > button {{ min-height:3rem; }}

.trace-grid {{ display:grid; grid-template-columns:155px 1fr; gap:.36rem .75rem; font-size:.84rem; }}
.trace-label {{ color:var(--axiz-muted); }}
.trace-value {{ overflow-wrap:anywhere; color:#c7d6df; }}
.tech-pill {{ display:inline-block; border:1px solid #243d4e; background:#0d1d29; border-radius:999px; padding:.2rem .5rem; color:#86a0b2 !important; font-size:.68rem; margin:.12rem .18rem .12rem 0; }}
.settings-note {{ border:1px solid #1f3545; background:#0a1822; border-radius:10px; padding:.6rem .66rem; color:#7991a3; font-size:.71rem; line-height:1.45; }}

/* Inline composer is a normal flex row below the message viewport. It is not
   absolute/fixed, so its full border and controls always remain inside the
   viewport while the message region flexes to the remaining height. */
.st-key-center_shell div[data-testid="stElementContainer"]:has(.st-key-chat_composer) {{
  flex:0 0 auto !important;
  min-height:0 !important;
  margin:.55rem 0 0 !important;
  padding:.18rem 0 .22rem !important;
  background:var(--axiz-bg) !important;
  overflow:visible !important;
}}
.st-key-chat_composer {{
  width:100%;
  max-width:{CHAT_MAX_WIDTH}px;
  margin:0 auto;
  position:relative !important;
  z-index:20 !important;
}}
.st-key-chat_composer [data-testid="stChatInput"] {{ background:var(--axiz-card) !important; border:1px solid var(--axiz-line-strong) !important; border-radius:16px !important; box-shadow:0 12px 30px rgba(0,0,0,.26) !important; padding:.28rem .38rem .28rem .82rem; }}
.st-key-chat_composer [data-testid="stChatInput"]:focus-within {{ border-color:#3a9cbc !important; box-shadow:0 0 0 1px rgba(67,195,236,.24),0 14px 32px rgba(0,0,0,.3) !important; }}
.st-key-chat_composer [data-testid="stChatInput"] > div,
.st-key-chat_composer [data-testid="stChatInput"] [data-baseweb="textarea"],
.st-key-chat_composer [data-testid="stChatInput"] [data-baseweb="base-input"] {{ background:transparent !important; border:0 !important; box-shadow:none !important; }}
.st-key-chat_composer [data-testid="stChatInput"] textarea {{ min-height:52px; max-height:128px; background:transparent !important; border:0 !important; box-shadow:none !important; color:#e3edf4 !important; caret-color:var(--axiz-accent); resize:none; }}
.st-key-chat_composer [data-testid="stChatInput"] textarea::placeholder {{ color:#61798b !important; opacity:1; }}
.st-key-chat_composer [data-testid="stChatInput"] button {{ border:1px solid #2d6f87 !important; border-radius:11px !important; background:#14384a !important; color:#8ce5ff !important; }}

/* Narrow windows fall back to normal document flow so controls remain usable. */
@media(max-width:1050px) {{
  html, body, .stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"] {{ height:auto !important; max-height:none !important; overflow:auto !important; }}
  [data-testid="stMainBlockContainer"],.block-container {{ height:auto !important; max-height:none !important; overflow:visible !important; }}
  .st-key-left_nav_panel,.st-key-right_settings_panel {{ height:auto !important; max-height:none !important; overflow:visible !important; }}
  .st-key-chat_scroll_panel {{ flex:0 0 auto !important; height:min(58dvh,520px) !important; max-height:min(58dvh,520px) !important; min-height:300px !important; overflow-y:auto !important; }}
  .st-key-center_shell div[data-testid="stElementContainer"]:has(.st-key-chat_composer) {{ padding:.4rem 0 .2rem !important; background:transparent !important; }}
}}
</style>
    """
)



def new_conversation() -> str:
    conversation_id = str(uuid4())
    now = datetime.now(UTC)
    st.session_state.conversations[conversation_id] = {
        "id": conversation_id,
        "title": "Nueva conversación",
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "last_payment_id": None,
    }
    st.session_state.current_conversation_id = conversation_id
    return conversation_id


def current_conversation() -> dict[str, Any]:
    conversation_id = st.session_state.current_conversation_id
    if not conversation_id or conversation_id not in st.session_state.conversations:
        conversation_id = new_conversation()
    return st.session_state.conversations[conversation_id]


def add_message(role: str, content: str, payload: dict[str, Any] | None = None) -> None:
    conversation = current_conversation()
    conversation["messages"].append({"role": role, "content": content, "payload": payload})
    conversation["updated_at"] = datetime.now(UTC)
    if role == "user" and conversation["title"] == "Nueva conversación":
        conversation["title"] = conversation_title(content)


def delete_conversation(conversation_id: str) -> None:
    conversations = st.session_state.conversations
    conversations.pop(conversation_id, None)
    if not conversations:
        new_conversation()
        return
    if st.session_state.current_conversation_id == conversation_id:
        newest = max(conversations.values(), key=lambda item: item["updated_at"])
        st.session_state.current_conversation_id = newest["id"]


def clear_current_conversation() -> None:
    conversation = current_conversation()
    conversation["messages"] = []
    conversation["last_payment_id"] = None
    conversation["title"] = "Nueva conversación"
    conversation["updated_at"] = datetime.now(UTC)


def render_brand_header() -> None:
    left, center, right = st.columns([1, 6, 1])
    del left, right
    with center:
        st.image(AXIZ_LOGO, width=170)
    st.markdown(
        "<div class='sidebar-brand'>GraphRAG Payments · Investigación de pagos</div>",
        unsafe_allow_html=True,
    )


def render_left_navigation(service_ready: bool, readiness: dict[str, Any]) -> None:
    with st.container(key="left_nav_panel"):
        head_left, head_right = st.columns([0.78, 0.22], vertical_alignment="center")
        with head_left:
            st.markdown("<div class='panel-title'>Conversaciones</div>", unsafe_allow_html=True)
        with head_right:
            if st.button("‹", key="collapse-left", help="Ocultar historial", width="stretch"):
                st.session_state.left_sidebar_collapsed = True
                st.rerun()

        render_brand_header()

        status_class = "ready" if service_ready else "offline"
        status_text = "API y Neo4j disponibles" if service_ready else "API / Neo4j no disponible"
        st.markdown(
            f"<div class='left-status-card {status_class}'>"
            "<span class='left-status-indicator'></span>"
            f"<span>{status_text}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='left-tech-row'>"
            "<span class='tech-pill'>Vector</span>"
            "<span class='tech-pill'>Full-text</span>"
            "<span class='tech-pill'>Graph</span>"
            "<span class='tech-pill'>GraphSAGE</span>"
            "<span class='tech-pill'>SSE</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        provider = str(readiness.get("generation_provider", "unknown"))
        model = str(readiness.get("generation_model", "—"))
        key_configured = bool(readiness.get("openai_key_configured", False))
        runtime = f"OpenAI · {model}" if provider == "openai" else "Deterministic"
        if provider == "openai" and not key_configured:
            runtime += " · API key ausente"
        st.caption(f"Generación activa: **{runtime}**")
        with st.expander("GraphSAGE · administración y pruebas", expanded=False):
            st.caption(
                "Entrena el modelo o busca pagos similares."
            )
            if service_ready:
                try:
                    neural = ApiClient().graphsage_status()
                    st.caption(
                        f"Embeddings: {neural['embedded_payments']}/{neural['payment_count']} "
                        f"pagos · {'listo' if neural['ready'] else 'sin entrenar'}"
                    )
                except httpx.HTTPError as exc:
                    st.warning(f"Estado GraphSAGE no disponible: {exc}")
                epochs = st.slider("Épocas de entrenamiento", 1, 20, 5, key="sage-epochs")
                dimension = st.select_slider(
                    "Dimensión embedding", options=[8, 16, 32, 64], value=32,
                    key="sage-dimension",
                )
                if st.button("Entrenar GraphSAGE", key="sage-train", width="stretch"):
                    with st.spinner("Entrenando GraphSAGE sobre Neo4j GDS…"):
                        try:
                            result = ApiClient().graphsage_train(epochs, dimension)
                            st.success(
                                f"Modelo entrenado: {result['embedded_payments']} pagos, "
                                f"{result['embedding_dimension']} dimensiones."
                            )
                        except httpx.HTTPError as exc:
                            st.error(f"No se pudo entrenar GraphSAGE: {exc}")
                payment_id = st.text_input(
                    "ID de pago", value="PAY-1007", key="sage-payment",
                )
                if st.button("Buscar pagos similares", key="sage-search", width="stretch"):
                    try:
                        matches = ApiClient().graphsage_similar(payment_id.strip())
                        if matches["neighbors"]:
                            st.dataframe(matches["neighbors"], hide_index=True)
                        else:
                            st.info("No se encontraron vecinos con el filtro actual.")
                    except httpx.HTTPError as exc:
                        st.error(f"Consulta neuronal fallida: {exc}")
                st.caption(
                    "Similitud ≠ causa común."
                )
        if st.button("＋ Nuevo chat", type="primary", width="stretch"):
            new_conversation()
            st.rerun()

        search = st.text_input(
            "Buscar chats",
            placeholder="Buscar conversaciones",
            label_visibility="collapsed",
            key="chat-search",
        )

        ordered_conversations = sorted(
            st.session_state.conversations.values(),
            key=lambda item: item["updated_at"],
            reverse=True,
        )
        filtered = [
            item for item in ordered_conversations if search.lower() in item["title"].lower()
        ]
        groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict(
            (name, [])
            for name in ("Hoy", "Ayer", "Últimos 7 días", "Últimos 30 días", "Anteriores")
        )
        for item in filtered:
            groups[conversation_group(item["updated_at"])].append(item)

        for group_name, conversations in groups.items():
            if not conversations:
                continue
            st.markdown(
                f"<div class='session-group'>{group_name}</div>",
                unsafe_allow_html=True,
            )
            for item in conversations:
                conversation_id = item["id"]
                active = conversation_id == st.session_state.current_conversation_id
                title_col, menu_col = st.columns(
                    [0.82, 0.18], gap="small", vertical_alignment="center"
                )
                with title_col:
                    if st.button(
                        f"{'● ' if active else ''}{item['title']}",
                        key=f"conversation-{conversation_id}",
                        type="primary" if active else "secondary",
                        width="stretch",
                        help=item["title"],
                    ):
                        st.session_state.current_conversation_id = conversation_id
                        st.rerun()
                with menu_col:
                    with st.popover("⋯"):
                        st.caption("Opciones del chat")
                        with st.form(f"rename-{conversation_id}"):
                            new_title = st.text_input(
                                "Nombre",
                                value=item["title"],
                                key=f"rename-title-{conversation_id}",
                            )
                            rename = st.form_submit_button("Renombrar", width="stretch")
                        if rename and new_title.strip():
                            item["title"] = conversation_title(new_title.strip(), max_length=48)
                            item["updated_at"] = datetime.now(UTC)
                            st.rerun()
                        if st.button(
                            "Eliminar chat",
                            key=f"delete-{conversation_id}",
                            width="stretch",
                        ):
                            delete_conversation(conversation_id)
                            st.rerun()
                st.markdown(
                    f"<div class='session-caption'>{len(item['messages'])} mensajes</div>",
                    unsafe_allow_html=True,
                )


def render_right_settings() -> None:
    with st.container(key="right_settings_panel"):
        st.markdown("<div class='panel-title'>Configuración</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-subtitle'>Ajustes del agente.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("**Recuperación**")
        st.selectbox(
            "Modo",
            options=["Neural GraphRAG inteligente", "GraphRAG tradicional"],
            key="retrieval_mode",
            help=(
                "Inteligente: combina GraphRAG con GraphSAGE al pedir pagos similares o relacionados. "
                "Tradicional: solo búsqueda híbrida y relaciones del grafo."
            ),
        )
        if st.session_state.retrieval_mode == "Neural GraphRAG inteligente":
            st.caption("Con un ID: similares/relacionados → +GraphSAGE.")
            reference = current_conversation().get("last_payment_id")
            st.caption(f"Pago actual: **{reference or '—'}**")
        st.markdown("**Búsqueda**")
        st.session_state.top_k = st.slider(
            "Contextos (Top K)",
            min_value=1,
            max_value=10,
            value=int(st.session_state.top_k),
            help="Cantidad máxima de contextos entregados al generador.",
        )
        st.session_state.show_trace = st.toggle(
            "Actividad técnica",
            value=bool(st.session_state.show_trace),
            help="Muestra índices, expansión del grafo y trazabilidad del retrieval.",
        )
        st.session_state.show_evidence = st.toggle(
            "Evidencia recuperada",
            value=bool(st.session_state.show_evidence),
            help="Permite inspeccionar chunks, códigos y pagos recuperados.",
        )
        st.session_state.show_query_progress = st.toggle(
            "Progreso de consulta",
            value=bool(st.session_state.show_query_progress),
            help="Muestra las etapas visibles de recuperación mientras se procesa la pregunta.",
        )

        st.divider()
        st.markdown("**Conversación**")
        if st.button("Limpiar conversación", key="clear-current", width="stretch"):
            clear_current_conversation()
            st.rerun()

        st.markdown(
            "<div class='settings-note'>"
            "Historial disponible durante esta sesión."
            "</div>",
            unsafe_allow_html=True,
        )


def render_topbar(
    conversation: dict[str, Any], ready: bool, readiness: dict[str, Any]
) -> None:
    title = escape(conversation["title"] or "Nueva conversación")
    dot_class = "axiz-dot" if ready else "axiz-dot offline"
    status = "API y Neo4j disponibles" if ready else "API / Neo4j no disponible"
    provider = str(readiness.get("generation_provider", "unknown"))
    model = str(readiness.get("generation_model", "—"))
    generation_chip = f"OpenAI · {escape(model)}" if provider == "openai" else "Deterministic"
    st.markdown(
        f"""
        <div class="axiz-topbar">
          <div>
            <h1>{title}</h1>
            <div class="status"><span class="{dot_class}"></span>{status}</div>
          </div>
          <div class="axiz-chips">
            <span class="axiz-chip">GraphRAG</span>
            <span class="axiz-chip">Hybrid retrieval</span>
            <span class="axiz-chip">SSE</span>
            <span class="axiz-chip">{generation_chip}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_contexts(contexts: list[dict[str, Any]]) -> None:
    if not contexts:
        st.caption("La consulta no devolvió contextos para mostrar.")
        return

    for index, context in enumerate(contexts, start=1):
        title = context.get("title", f"Contexto {index}")
        score = context.get("score")
        score_text = f" · score `{float(score):.4f}`" if score is not None else ""
        st.markdown(f"**{index}. {title}**{score_text}")
        st.caption(context.get("text", ""))

        reasons = context.get("reason_codes") or []
        if reasons:
            codes = ", ".join(str(item.get("code", "—")) for item in reasons)
            st.markdown(f"**Códigos relacionados:** {codes}")

        payments = context.get("related_payments") or []
        if payments:
            st.dataframe(payments, width="stretch", hide_index=True)


def render_trace(trace: dict[str, Any] | None) -> None:
    rows = trace_rows(trace)
    html = ["<div class='trace-grid'>"]
    for label, value in rows:
        html.append(
            f"<div class='trace-label'>{escape(label)}</div>"
            f"<div class='trace-value'>{escape(value)}</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_assistant_payload(payload: dict[str, Any]) -> None:
    # Always explain whether the neural model actually ran, including when
    # retrieval is disabled, the question does not request it, or GDS is down.
    trace = payload.get("trace") or {}
    usage = graphsage_usage(trace)
    if trace.get("neural_route") == "unavailable":
        st.warning(usage)
    else:
        st.caption(usage)
    if not st.session_state.show_trace and not st.session_state.show_evidence:
        return
    with st.expander("Actividad técnica de GraphRAG", expanded=False):
        timings = payload.get("timings") or {}
        if timings:
            st.caption(
                "SSE · retrieval "
                f"{timings.get('retrieval_ms', '—')} ms · generación "
                f"{timings.get('generation_ms', '—')} ms · total "
                f"{timings.get('total_ms', '—')} ms"
            )
        if st.session_state.show_trace and st.session_state.show_evidence:
            trace_tab, evidence_tab = st.tabs(["Trazabilidad", "Evidencia recuperada"])
            with trace_tab:
                st.caption(
                    "Muestra el mecanismo técnico de recuperación ejecutado por la PoC "
                    "sin exponer razonamiento privado del modelo."
                )
                render_trace(payload.get("trace"))
            with evidence_tab:
                render_contexts(payload.get("contexts", []))
        elif st.session_state.show_trace:
            render_trace(payload.get("trace"))
        else:
            render_contexts(payload.get("contexts", []))
        if st.session_state.show_evidence and payload.get("neural_neighbors"):
            st.markdown("**Vecinos estructurales GraphSAGE**")
            st.caption(
                "Estos pagos presentan similitud de embeddings; no se confirma "
                "que compartan la misma causa del incidente."
            )
            st.dataframe(payload["neural_neighbors"], hide_index=True, width="stretch")


def render_benchmark_report(report: dict[str, Any], report_id: str) -> None:
    """Render the experiment within the full-width chat message, not the sidebar."""
    k = int(report["top_k"])
    st.caption(f"Datos ficticios · {report['cohort_payments']} pagos · {report['queries']} consultas · K={k}")
    rules_col, sage_col, new_col = st.columns(3)
    rules_col.metric("Recall reglas", f"{report['baseline']['recall_at_k']:.1%}")
    sage_col.metric("Recall GraphSAGE", f"{report['graphsage']['recall_at_k']:.1%}")
    new_col.metric("Hallazgos extra", report["additional_relevant_hits"])
    st.dataframe(strategy_rows(report), hide_index=True, width="stretch")
    st.caption("*La unión usa hasta 2K candidatos; no es una comparación con igual presupuesto.")
    st.caption(
        f"GraphSAGE aportó relevantes fuera del top-{k} de reglas en "
        f"{report['queries_with_additional_relevant']}/{report['queries']} consultas "
        f"({report['distinct_additional_relevant']} pagos distintos)."
    )
    extra_rows = additional_payment_rows(report)
    if extra_rows:
        with st.expander("Pagos relevantes adicionales · GraphSAGE", expanded=True):
            st.dataframe(extra_rows, hide_index=True, width="stretch", height=min(38 * len(extra_rows) + 38, 230))
    else:
        st.caption("Sin pagos relevantes adicionales en esta comparación.")
    for item in report["results"]:
        additional = item["additional_relevant_ids"]
        with st.expander(f"{item['anchor']} · {item['incident']} · +{len(additional)}", expanded=False):
            if additional:
                st.success("Solo GraphSAGE en top-K: " + ", ".join(additional))
            else:
                st.caption("Sin relevantes adicionales en top-K.")
            st.dataframe(query_rows(item), hide_index=True, width="stretch")
    with st.expander("Alcance y límites", expanded=False):
        st.caption(
            "Etiquetas ficticias, ajenas al entrenamiento; solo se compara recuperación de pagos. "
            "No demuestra causa raíz, fraude ni superioridad sobre datos reales."
        )
    st.download_button(
        "Exportar JSON", data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="graphsage_value_benchmark.json", mime="application/json",
        key=f"benchmark-download-{report_id}",
    )
    st.caption("Otro K: escribe /evaluar graphsage k=10 (1–20).")


def render_benchmark_assistant(client: ApiClient, top_k: int) -> None:
    """Run only the actual benchmark endpoint, inside the visible chat thread."""
    st.html('<div class="stream-active-marker" aria-hidden="true"></div>')
    with st.chat_message("assistant", avatar=APP_ICON):
        report: dict[str, Any] | None = None
        with st.status("Comparando reglas y GraphSAGE…", expanded=False) as progress:
            try:
                report = client.graphsage_value_benchmark(top_k)
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail")
                except (ValueError, AttributeError, TypeError):
                    detail = None
                if exc.response.status_code == 409:
                    message = f"No se pudo ejecutar la evaluación: {detail or 'Carga los datos BENCH- y entrena GraphSAGE.'}"
                elif exc.response.status_code >= 500:
                    message = "La API falló durante la evaluación. Revisa los logs de api."
                else:
                    message = f"No se pudo ejecutar la evaluación (HTTP {exc.response.status_code}): {detail or 'Error de la API.'}"
                progress.update(label="Evaluación no completada", state="error")
                st.error(message)
                add_message("assistant", message)
            except httpx.HTTPError:
                message = "No se pudo conectar con la API. Verifica los servicios."
                progress.update(label="Evaluación no completada", state="error")
                st.error(message)
                add_message("assistant", message)
            except Exception:
                LOGGER.exception("Benchmark chat execution failed")
                message = "Error al procesar la evaluación. Consulta los logs del frontend."
                progress.update(label="Evaluación no completada", state="error")
                st.error(message)
                add_message("assistant", message)
            else:
                progress.update(label="Comparación completada", state="complete")
            finally:
                st.session_state.pending_request = None
                st.session_state.pending_conversation_payment_id = None
                st.session_state.pending_conversation_id = None
                st.session_state.pending_benchmark_k = None
                st.session_state.pending_turn_kind = "chat"
        # Render outside the collapsed status widget. The full-width chat card
        # persists as an assistant message, without re-running the API on rerun.
        if report is not None:
            report_id = uuid4().hex
            content = "Resultado de la comparación de pagos (datos ficticios):"
            st.markdown(content)
            render_benchmark_report(report, report_id)
            add_message("assistant", content, {"kind": "benchmark", "report": report, "report_id": report_id})
    st.rerun()


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    avatar: Any = APP_ICON if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))
        payload = message.get("payload")
        if payload:
            if payload.get("kind") == "benchmark":
                render_benchmark_report(payload["report"], payload["report_id"])
            else:
                render_assistant_payload(payload)


def queue_chat_question(widget_key: str) -> None:
    """Persist chat input in the submit callback before the widget clears it."""
    value = st.session_state.get(widget_key)
    if isinstance(value, str) and value.strip():
        st.session_state.submitted_chat_question = value.strip()
        LOGGER.info("Chat question submitted; queued for dispatch")


def queue_example_question(question: str) -> None:
    """Queue example in widget callback before Streamlit reruns the page."""
    st.session_state.pending_question = question
    LOGGER.info("Example question selected; queued for dispatch")


def render_empty_state() -> None:
    # Keep the whole welcome surface under one keyed wrapper. During the first
    # streaming run CSS can hide this exact wrapper immediately, avoiding the
    # stale welcome screen that Streamlit may otherwise keep until the run ends.
    with st.container(key="empty_state"):
        st.markdown("<div class='hero-spacer'></div>", unsafe_allow_html=True)
        left, center, right = st.columns([3, 1, 3])
        del left, right
        with center:
            st.image(APP_ICON, width=70)
        st.markdown(
            """
            <div class="hero-title" style="text-align:center">¿Qué quieres investigar sobre tus pagos?</div>
            <div class="hero-subtitle" style="text-align:center">
              Consulta rechazos, incidencias y controles. La PoC combina recuperación semántica,
              búsqueda full-text y expansión del grafo para aportar contexto conectado a la respuesta.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='suggestion-label'>Preguntas de ejemplo</div>", unsafe_allow_html=True)
        with st.container(key="example_questions"):
            left_column, right_column = st.columns(2, gap="small")
            for index, question in enumerate((*EXAMPLE_QUESTIONS, BENCHMARK_EXAMPLE_QUESTION)):
                column = left_column if index % 2 == 0 else right_column
                with column:
                    st.button(
                        question, key=f"example-{index}", width="stretch",
                        on_click=queue_example_question, args=(question,),
                    )



_SCROLL_DRIVER_JS = r"""
(() => {
  const NONCE = "__NONCE__";
  let W = window;
  try { if (window.parent && window.parent.document) W = window.parent; } catch (e) {}
  const doc = W.document;
  const PANE = '.st-key-chat_scroll_panel';

  // One driver at a time (the script re-runs on every new question).
  try { if (W.__axizChat && W.__axizChat.stop) W.__axizChat.stop(); } catch (e) {}

  let pane = null, stick = true, raf = 0, lastTop = 0;
  const dist = () => pane.scrollHeight - pane.clientHeight - pane.scrollTop;
  const onScroll = () => {
    // Scrolling up releases the follow immediately; coming back to the bottom
    // re-arms it. Content growth alone never fires this with a smaller top.
    if (pane.scrollTop < lastTop - 2) stick = false;
    if (dist() < 48) stick = true;
    lastTop = pane.scrollTop;
  };
  const fit = () => {
    const root = doc.documentElement;
    const comp = doc.querySelector('.st-key-chat_composer');
    if (!pane || !comp || W.innerWidth <= 1050) { root.style.removeProperty('--axiz-chat-h'); return; }
    // Only inputs that do NOT depend on the pane's own height (no feedback):
    // where the pane starts and how tall the composer currently is.
    const top = pane.getBoundingClientRect().top;
    const composer = comp.getBoundingClientRect().height;
    const h = Math.floor(W.innerHeight - top - composer - 52);
    if (h >= 220) root.style.setProperty('--axiz-chat-h', h + 'px');
  };
  const tick = () => {
    raf = 0;
    const current = doc.querySelector(PANE);
    if (!current) return;
    if (current !== pane) {
      if (pane) pane.removeEventListener('scroll', onScroll);
      pane = current;
      pane.addEventListener('scroll', onScroll, {passive: true});
      stick = true;
    }
    fit();
    if (stick) { pane.scrollTop = pane.scrollHeight; lastTop = pane.scrollTop; }
  };
  const schedule = () => { if (!raf) raf = W.requestAnimationFrame(tick); };

  const observer = new MutationObserver(schedule);
  observer.observe(doc.body, {childList: true, subtree: true, characterData: true});
  W.addEventListener('resize', schedule);
  const timer = W.setInterval(schedule, 400);

  W.__axizChat = {
    nonce: NONCE,
    stop: () => {
      observer.disconnect();
      W.removeEventListener('resize', schedule);
      W.clearInterval(timer);
      if (pane) pane.removeEventListener('scroll', onScroll);
    },
  };
  schedule();
})();
"""


def render_scroll_driver(conversation_id: str) -> None:
    """Install the chat scroll driver (follow-while-streaming + viewport fit).

    The script text only changes when ``scroll_nonce`` or the conversation
    changes (new question / other chat), so Streamlit re-executes it exactly
    then, which re-arms "stick to bottom". While armed it follows the stream;
    the first upward scroll by the user releases it until they return to the
    bottom. It is rendered BEFORE the viewport so it is already running while
    the SSE run blocks the script.
    """
    nonce = f"{conversation_id}:{st.session_state.scroll_nonce}"
    script = _SCROLL_DRIVER_JS.replace("__NONCE__", nonce)
    with st.container(key="stream_scroll_driver"):
        if "unsafe_allow_javascript" in inspect.signature(st.html).parameters:
            # Runs in the main document: no iframe sandbox/origin involved.
            st.html(f"<script>{script}</script>", unsafe_allow_javascript=True)
        else:
            st.iframe(
                f"<!doctype html><html><body><script>{script}</script></body></html>",
                height=1,
                width=1,
                tab_index=-1,
            )


def render_streaming_assistant(
    client: ApiClient, question: str,
    *, retrieval_mode: str = "traditional", conversation_payment_id: str | None = None,
) -> None:
    """Stream the original retrieval or conversational automatic GraphSAGE route."""
    # This marker immediately suppresses any stale empty-state DOM left from
    # the previous render while the long-lived SSE run is still in progress.
    st.html('<div class="stream-active-marker" aria-hidden="true"></div>')
    with st.chat_message("assistant", avatar=APP_ICON):
        stream_state: dict[str, Any] = {"payload": None}

        # Busy feedback is ALWAYS visible, even when technical progress details
        # are disabled. Both modes use the same stream/UI lifecycle.
        initial = (
            "Iniciando Neural GraphRAG · conectando SSE…"
            if retrieval_mode == "auto" else "Iniciando GraphRAG · conectando SSE…"
        )
        progress = st.status(initial, expanded=False)

        def text_deltas():
            pending_parts: list[str] = []
            pending_chars = 0

            for message in client.query_stream(
                question, int(st.session_state.top_k),
                retrieval_mode=retrieval_mode,
                conversation_payment_id=conversation_payment_id,
            ):
                event = message.get("event")
                data = message.get("data") or {}

                if event == "stage":
                    if progress is not None:
                        progress.update(label=str(data.get("message", "Procesando…")))
                elif event == "retrieval":
                    if st.session_state.show_query_progress:
                        retrieval_ms = data.get("retrieval_ms", "—")
                        progress.write(f"Recuperación: {retrieval_ms} ms")
                        progress.write(graphsage_usage(data.get("trace") or {}))
                elif event == "delta":
                    delta = str(data.get("delta", ""))
                    if not delta:
                        continue
                    pending_parts.append(delta)
                    pending_chars += len(delta)
                    if pending_chars >= STREAM_RENDER_BATCH_CHARS or "\n" in delta:
                        yield "".join(pending_parts)
                        pending_parts.clear()
                        pending_chars = 0
                elif event == "complete":
                    if pending_parts:
                        yield "".join(pending_parts)
                        pending_parts.clear()
                        pending_chars = 0
                    stream_state["payload"] = data
                    if progress is not None:
                        timings = data.get("timings") or {}
                        total_ms = timings.get("total_ms", "—")
                        progress.update(
                            label=f"Respuesta completada · {total_ms} ms",
                            state="complete",
                            expanded=False,
                        )
                elif event == "error":
                    raise RuntimeError(
                        str(data.get("detail") or data.get("message") or "SSE error")
                    )

            if pending_parts:
                yield "".join(pending_parts)

        try:
            answer = st.write_stream(text_deltas(), cursor="▌")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("El stream SSE terminó sin contenido de respuesta.")

            payload = stream_state.get("payload")
            if not isinstance(payload, dict):
                payload = {
                    "answer": answer,
                    "generation_provider": "unknown",
                    "contexts": [],
                    "trace": {},
                }
            render_assistant_payload(payload)
            add_message("assistant", answer.strip(), payload)
        except Exception as exc:
            LOGGER.exception("Chat generation failed")
            error_message = (
                "No fue posible completar la consulta GraphRAG por streaming. "
                "Verifica el estado de `api`, Neo4j y la configuración del proveedor de generación."
            )
            if progress is not None:
                progress.update(label="Consulta GraphRAG fallida", state="error", expanded=False)
            st.error(error_message)
            st.caption(str(exc))
            add_message("assistant", error_message)
        finally:
            st.session_state.pending_request = None
            st.session_state.pending_conversation_payment_id = None
            st.session_state.pending_conversation_id = None

    st.rerun()


def render_chat_area(
    client: ApiClient,
    conversation: dict[str, Any],
    service_ready: bool,
    readiness: dict[str, Any],
) -> str | None:
    render_topbar(conversation, service_ready, readiness)

    pending_request = st.session_state.get("pending_request")
    belongs_here = (
        bool(pending_request)
        and st.session_state.get("pending_conversation_id") == conversation["id"]
    )
    render_scroll_driver(conversation["id"])
    with st.container(border=False, key="chat_scroll_panel"):
        with st.container(border=False, key="chat_thread"):
            messages = conversation["messages"]
            if messages:
                for message in messages:
                    render_message(message)
            elif not belongs_here:
                render_empty_state()
            # The assistant MUST live inside the scrolling thread. Previously the
            # first reply was drawn outside the viewport and was invisible while
            # a slow neural lookup blocked the Streamlit run.
            assistant_slot = st.empty() if belongs_here else None
        st.html('<div class="chat-bottom-anchor" aria-hidden="true"></div>')

    # Draw the composer before blocking on the network, so the busy/disabled
    # state is immediately visible for both modes and the chat layout is stable.
    with st.container(key="chat_composer"):
        widget_key = f"chat-input-{conversation['id']}"
        question = st.chat_input(
            "Pregunta sobre rechazos, incidencias o controles de payment processing",
            disabled=belongs_here,
            key=widget_key,
            on_submit=queue_chat_question,
            args=(widget_key,),
        )

    if belongs_here and assistant_slot is not None:
        with assistant_slot.container():
            if st.session_state.pending_turn_kind == "benchmark":
                render_benchmark_assistant(client, int(st.session_state.pending_benchmark_k))
            else:
                render_streaming_assistant(
                    client, str(pending_request),
                    retrieval_mode=st.session_state.pending_retrieval_mode,
                    conversation_payment_id=st.session_state.pending_conversation_payment_id,
                )
    # Callback normally captures the question at the start of the run; this
    # fallback also supports Streamlit releases whose callback semantics differ.
    return question


def start_queued_turn(question: str) -> None:
    """Commit the user message BEFORE rendering, then stream from its viewport."""
    if st.session_state.get("pending_request") or not question.strip():
        return
    conversation = current_conversation()
    benchmark_k = benchmark_top_k(question)
    if benchmark_k is not None:
        # Explicit benchmark question: never pass it to the LLM/SSE normal route.
        add_message("user", question.strip())
        st.session_state.pending_request = question.strip()
        st.session_state.pending_conversation_id = conversation["id"]
        st.session_state.pending_turn_kind = "benchmark"
        st.session_state.pending_benchmark_k = benchmark_k
        st.session_state.scroll_nonce += 1
        LOGGER.info("Dispatching actual GraphSAGE-vs-rules benchmark; top_k=%s", benchmark_k)
        return
    st.session_state.pending_turn_kind = "chat"
    st.session_state.pending_benchmark_k = None
    plan = plan_turn(conversation, question, st.session_state.retrieval_mode)
    add_message("user", plan["question"])
    st.session_state.pending_request = plan["question"]
    st.session_state.pending_conversation_id = conversation["id"]
    st.session_state.pending_conversation_payment_id = plan["conversation_payment_id"]
    st.session_state.pending_retrieval_mode = plan["retrieval_mode"]
    st.session_state.scroll_nonce += 1
    LOGGER.info(
        "Dispatching chat question; mode=%s; explicit_payment_count=%s",
        plan["retrieval_mode"], plan["explicit_payment_count"],
    )


if not st.session_state.conversations:
    new_conversation()

# Widget callbacks execute before the main script. Consume their pending text
# now: the user message and assistant placeholder must both be rendered within
# the scrolling chat thread before a potentially slow GraphSAGE/SSE request.
queued = st.session_state.pop("pending_question", None)
queued_chat = st.session_state.pop("submitted_chat_question", None)
if queued or queued_chat:
    start_queued_turn(str(queued or queued_chat))

client = ApiClient()

try:
    readiness_payload = client.ready()
    service_ready = readiness_payload.get("status") == "ready"
except httpx.HTTPError:
    readiness_payload = {}
    service_ready = False

conversation = current_conversation()
submitted_question: str | None = None

if st.session_state.left_sidebar_collapsed:
    center_col, right_col = st.columns([1.0, 0.29], gap="large")
    with right_col:
        render_right_settings()
    with center_col:
        with st.container(key="center_shell"):
            with st.container(key="left_reopen_row"):
                if st.button("☰", key="open-left", help="Mostrar historial"):
                    st.session_state.left_sidebar_collapsed = False
                    st.rerun()
            submitted_question = render_chat_area(
                client, conversation, service_ready, readiness_payload
            )
else:
    left_col, center_col, right_col = st.columns([0.29, 0.94, 0.29], gap="large")
    with left_col:
        render_left_navigation(service_ready, readiness_payload)
    with right_col:
        render_right_settings()
    with center_col:
        with st.container(key="center_shell"):
            submitted_question = render_chat_area(
                client, conversation, service_ready, readiness_payload
            )

if submitted_question and not st.session_state.get("pending_request"):
    # Fallback for Streamlit versions that expose st.chat_input's return value
    # without executing on_submit. Re-render before starting network I/O.
    start_queued_turn(submitted_question)
    st.rerun()
