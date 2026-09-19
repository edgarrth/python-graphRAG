from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import streamlit as st
from api_client import ApiClient
from ui_helpers import EXAMPLE_QUESTIONS, conversation_group, conversation_title, trace_rows

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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
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
  --axiz-accent-soft:#112b38;
  --axiz-success:#4bd8a0;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background:var(--axiz-bg) !important;
  color:var(--axiz-text);
}

html, body { color-scheme:dark; }
.stApp {
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
[data-testid="stHeader"] {
  background: transparent;
  border: none;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stMainMenu"],
.stDeployButton,
footer {
  display:none !important;
}

[data-testid="stSidebar"] {
  min-width:286px;
  max-width:286px;
  background:var(--axiz-sidebar);
  border-right:1px solid var(--axiz-line);
}
[data-testid="stSidebar"] * { color:var(--axiz-text); }
[data-testid="stSidebar"] .stTextInput input {
  background:#07121b;
  border-color:#294154;
  color:var(--axiz-text);
}
[data-testid="stSidebar"] .stButton button {
  border:1px solid #203747;
  border-radius:9px;
  background:#0e1d29;
  color:#cbd9e3;
  text-align:left;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background:#14384a;
  border-color:#2c91b0;
  color:#8ce5ff;
}
[data-testid="stSidebar"] hr { border-color:var(--axiz-line); }

.block-container {
  max-width:1040px;
  padding-top:1.05rem;
  padding-bottom:8.8rem;
}
h1,h2,h3,h4 { color:#edf5fa !important; letter-spacing:-.015em; }
p,li,label,[data-testid="stCaptionContainer"] { color:#b7c8d4; }
a { color:var(--axiz-accent); }

.sidebar-brand {
  color:#f0f7fb;
  font-size:.82rem;
  font-weight:720;
  text-align:center;
  margin:.12rem 0 .7rem;
}
.session-group {
  color:#667f92 !important;
  font-size:.68rem;
  font-weight:750;
  letter-spacing:.08em;
  margin:.9rem 0 .25rem;
  text-transform:uppercase;
}
.session-caption {
  color:#587084 !important;
  font-size:.65rem;
  margin:-.42rem 0 .28rem .35rem;
}

.axiz-topbar {
  display:flex;
  justify-content:space-between;
  gap:18px;
  align-items:flex-end;
  padding:.1rem 0 1rem;
  border-bottom:1px solid var(--axiz-line);
  margin-bottom:1.25rem;
}
.axiz-topbar h1 {
  margin:0;
  font-size:1.16rem;
  line-height:1.25;
}
.axiz-topbar .status {
  margin-top:.35rem;
  color:#728b9e;
  font-size:.72rem;
}
.axiz-dot {
  display:inline-block;
  width:7px;
  height:7px;
  border-radius:50%;
  background:var(--axiz-success);
  box-shadow:0 0 9px rgba(75,216,160,.45);
  margin-right:7px;
}
.axiz-dot.offline {
  background:#748797;
  box-shadow:none;
}
.axiz-chips {
  display:flex;
  gap:7px;
  flex-wrap:wrap;
  justify-content:flex-end;
}
.axiz-chip {
  border:1px solid #243d4e;
  background:#0d1d29;
  color:#86a0b2;
  border-radius:999px;
  padding:.35rem .62rem;
  font:.68rem ui-monospace,SFMono-Regular,Consolas,monospace;
}

.hero-spacer { height:8vh; }
.hero-title {
  color:#edf5fa;
  font-size:1.78rem;
  font-weight:780;
  letter-spacing:-.035em;
  margin:.65rem 0 .35rem;
}
.hero-subtitle {
  color:#7890a3;
  font-size:.94rem;
  line-height:1.55;
  max-width:660px;
  margin:0 auto;
}
.suggestion-label {
  color:#667f92;
  font-size:.72rem;
  font-weight:750;
  letter-spacing:.06em;
  margin:1.35rem 0 .45rem;
  text-transform:uppercase;
}

/* ChatGPT-like conversation surface, retaining the reference project's dark Axiz theme. */
[data-testid="stChatMessage"] {
  background:transparent;
  border:0;
  padding:.45rem .15rem 1.05rem;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
  background:var(--axiz-card);
  border:1px solid var(--axiz-line);
  border-radius:13px;
  padding:1rem 1.1rem;
  box-shadow:0 14px 34px rgba(0,0,0,.14);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stChatMessageContent"] {
  background:var(--axiz-card-user);
  border-color:#323653;
  max-width:82%;
  margin-left:auto;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
  border:1px solid #29445a;
  border-radius:9px;
  background:#0f2230;
}

[data-testid="stExpander"], [data-testid="stStatusWidget"] {
  background:#0a1721;
  border:1px solid var(--axiz-line);
  border-radius:11px;
}
[data-testid="stDataFrame"] {
  border:1px solid var(--axiz-line);
  border-radius:10px;
  overflow:hidden;
}
.stButton>button[kind="primary"],
.stFormSubmitButton>button[kind="primary"] {
  background:var(--axiz-accent);
  border-color:var(--axiz-accent);
  color:#041018;
  font-weight:750;
}
.stButton>button {
  border-radius:9px;
  border-color:#294154;
  background:#101e2a;
  color:#c6d7e3;
}
.stTextInput input,.stTextArea textarea {
  background:#091721;
  border-color:#294154;
  color:#e4eef5;
}
.stTextInput input::placeholder,.stTextArea textarea::placeholder {
  color:#60788a;
  opacity:1;
}
.stTextInput input:focus,.stTextArea textarea:focus {
  border-color:#3a9cbc;
  box-shadow:0 0 0 1px #3a9cbc;
}

/* Suggested prompts should read like prompt cards, not generic form buttons. */
[data-testid="stMain"] .stButton > button {
  min-height:3.15rem;
  text-align:left;
}

.trace-grid {
  display:grid;
  grid-template-columns:165px 1fr;
  gap:.38rem .8rem;
  font-size:.86rem;
}
.trace-label { color:var(--axiz-muted); }
.trace-value { overflow-wrap:anywhere; color:#c7d6df; }
.tech-pill {
  display:inline-block;
  border:1px solid #243d4e;
  background:#0d1d29;
  border-radius:999px;
  padding:.22rem .55rem;
  color:#86a0b2 !important;
  font-size:.7rem;
  margin:.15rem .25rem .15rem 0;
}

/* Streamlit renders chat_input inside a fixed bottom portal. */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"],
.stBottom,
.stChatFloatingInputContainer {
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
}
[data-testid="stBottomBlockContainer"] {
  background:linear-gradient(180deg,rgba(8,16,24,0) 0%,rgba(8,16,24,.96) 28%,var(--axiz-bg) 100%) !important;
  padding-top:1.25rem;
  padding-bottom:1rem;
}
[data-testid="stChatInput"] {
  background:var(--axiz-card) !important;
  border:1px solid var(--axiz-line-strong) !important;
  border-radius:16px !important;
  box-shadow:0 16px 40px rgba(0,0,0,.32) !important;
  padding:.35rem .4rem .35rem .85rem;
  transition:border-color .16s ease, box-shadow .16s ease;
}
[data-testid="stChatInput"]:focus-within {
  border-color:#3a9cbc !important;
  box-shadow:0 0 0 1px rgba(67,195,236,.24),0 18px 42px rgba(0,0,0,.34) !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
}
[data-testid="stChatInput"] textarea {
  min-height:58px;
  background:transparent !important;
  border:0 !important;
  box-shadow:none !important;
  color:#e3edf4 !important;
  caret-color:var(--axiz-accent);
  resize:none;
}
[data-testid="stChatInput"] textarea::placeholder {
  color:#61798b !important;
  opacity:1;
}
[data-testid="stChatInput"] button {
  border:1px solid #2d6f87 !important;
  border-radius:11px !important;
  background:#14384a !important;
  color:#8ce5ff !important;
}
[data-testid="stChatInput"] button:hover:not(:disabled) {
  background:#19506a !important;
  border-color:#43c3ec !important;
}
[data-testid="stChatInput"] button:disabled {
  background:#11222d !important;
  border-color:#203847 !important;
  color:#607786 !important;
  opacity:1;
}

@media(max-width:800px) {
  [data-testid="stSidebar"] { min-width:240px; max-width:240px; }
  .axiz-topbar { align-items:flex-start; flex-direction:column; }
  .axiz-chips { justify-content:flex-start; }
  .block-container { padding-inline:1rem; padding-bottom:8rem; }
  [data-testid="stBottomBlockContainer"] { padding-inline:.75rem; }
  .trace-grid { grid-template-columns:1fr; }
}
</style>
    """,
    unsafe_allow_html=True,
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


def render_brand_header() -> None:
    left, center, right = st.columns([1, 7, 1])
    del left, right
    with center:
        st.image(AXIZ_LOGO, width=210)
    st.markdown(
        "<div class='sidebar-brand'>GraphRAG Payments · Investigación de pagos</div>",
        unsafe_allow_html=True,
    )


def render_topbar(conversation: dict[str, Any], ready: bool) -> None:
    title = escape(conversation["title"] or "Nueva conversación")
    dot_class = "axiz-dot" if ready else "axiz-dot offline"
    status = "API y Neo4j disponibles" if ready else "API / Neo4j no disponible"
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
            <span class="axiz-chip">Neo4j</span>
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
    if not st.session_state.show_trace:
        return
    with st.expander("Actividad técnica de GraphRAG", expanded=False):
        trace_tab, evidence_tab = st.tabs(["Trazabilidad", "Evidencia recuperada"])
        with trace_tab:
            st.caption(
                "Muestra el mecanismo técnico de recuperación ejecutado por la PoC "
                "sin exponer razonamiento privado del modelo."
            )
            render_trace(payload.get("trace"))
        with evidence_tab:
            render_contexts(payload.get("contexts", []))


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    avatar: Any = APP_ICON if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(message.get("content", ""))
        payload = message.get("payload")
        if payload:
            render_assistant_payload(payload)


def render_empty_state() -> None:
    st.markdown("<div class='hero-spacer'></div>", unsafe_allow_html=True)
    left, center, right = st.columns([3, 1, 3])
    del left, right
    with center:
        st.image(APP_ICON, width=72)
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
    st.markdown(
        "<div class='suggestion-label'>Preguntas de ejemplo</div>",
        unsafe_allow_html=True,
    )
    left_column, right_column = st.columns(2, gap="small")
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        column = left_column if index % 2 == 0 else right_column
        with column:
            if st.button(question, key=f"example-{index}", width="stretch"):
                st.session_state.pending_question = question


for key, default in {
    "conversations": {},
    "current_conversation_id": None,
    "pending_question": None,
    "show_trace": True,
    "top_k": 4,
}.items():
    st.session_state.setdefault(key, default)

if not st.session_state.conversations:
    new_conversation()

client = ApiClient()

try:
    readiness_payload = client.ready()
    service_ready = readiness_payload.get("status") == "ready"
except httpx.HTTPError:
    readiness_payload = {}
    service_ready = False

with st.sidebar:
    render_brand_header()

    if st.button("＋ Nuevo chat", type="primary", width="stretch"):
        new_conversation()
        st.rerun()

    search = st.text_input(
        "Buscar chats",
        placeholder="Buscar conversaciones",
        label_visibility="collapsed",
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
        st.markdown(f"<div class='session-group'>{group_name}</div>", unsafe_allow_html=True)
        for item in conversations:
            conversation_id = item["id"]
            active = conversation_id == st.session_state.current_conversation_id
            title_col, menu_col = st.columns([0.84, 0.16], gap="small", vertical_alignment="center")
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
                    if st.button("Eliminar chat", key=f"delete-{conversation_id}", width="stretch"):
                        delete_conversation(conversation_id)
                        st.rerun()
            st.markdown(
                f"<div class='session-caption'>{len(item['messages'])} mensajes</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    with st.expander("Configuración de la PoC", expanded=False):
        st.session_state.top_k = st.slider(
            "Top K de recuperación",
            min_value=1,
            max_value=10,
            value=int(st.session_state.top_k),
            help="Cantidad máxima de contextos que GraphRAG entrega al generador.",
        )
        st.session_state.show_trace = st.toggle(
            "Actividad técnica",
            value=st.session_state.show_trace,
            help="Muestra recuperación, índices, expansión y evidencia sin razonamiento privado.",
        )
        st.markdown(
            "<span class='tech-pill'>Vector search</span>"
            "<span class='tech-pill'>Full-text</span>"
            "<span class='tech-pill'>Graph expansion</span>",
            unsafe_allow_html=True,
        )

    if service_ready:
        st.caption("● API ready · Neo4j ready")
    else:
        st.caption("○ API / Neo4j no disponible")

conversation = current_conversation()
render_topbar(conversation, service_ready)

messages = conversation["messages"]
if messages:
    for message in messages:
        render_message(message)
else:
    render_empty_state()

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Pregunta sobre rechazos, incidencias o controles de payment processing")
question = question or pending

if question:
    add_message("user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=APP_ICON):
        try:
            with st.status("Analizando la pregunta con GraphRAG…", expanded=True) as status:
                status.write("Buscando conocimiento por similitud semántica y texto completo…")
                payload = client.query(question, int(st.session_state.top_k))
                status.write("Expandiendo entidades y relaciones conectadas en Neo4j…")
                status.update(
                    label="Contexto GraphRAG recuperado",
                    state="complete",
                    expanded=False,
                )

            answer = payload["answer"]
            st.markdown(answer)
            render_assistant_payload(payload)
            add_message("assistant", answer, payload)
        except httpx.HTTPError as exc:
            error_message = (
                "No fue posible consultar la API GraphRAG. "
                "Verifica que `api`, `dataset-loader` y Neo4j estén saludables."
            )
            st.error(error_message)
            st.caption(str(exc))
            add_message("assistant", error_message)

    st.rerun()
