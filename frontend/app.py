from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from api_client import ApiClient


def render_contexts(contexts: list[dict[str, Any]]) -> None:
    for index, context in enumerate(contexts, start=1):
        st.markdown(f"**{index}. {context['title']}** · score `{context['score']:.4f}`")
        st.caption(context["text"])
        reasons = context.get("reason_codes") or []
        if reasons:
            st.write("Códigos:", ", ".join(item["code"] for item in reasons))
        payments = context.get("related_payments") or []
        if payments:
            st.dataframe(payments, use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="Axiz GraphRAG Payments",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { min-width: 305px; max-width: 305px; }
      .block-container { max-width: 1120px; padding-top: 1.4rem; }
      .trace-card { border-left: 3px solid rgba(128,128,128,.35); padding-left: .8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

client = ApiClient()
st.session_state.setdefault("messages", [])

with st.sidebar:
    st.markdown("### Axiz GraphRAG Payments")
    st.caption("PoC técnica · recuperación híbrida + expansión de grafo")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=4)
    st.markdown("#### Consultas sugeridas")
    examples = [
        "¿Por qué se rechazan pagos con código 05 y qué debería revisar operaciones?",
        "¿Qué evidencia hay sobre fondos insuficientes y qué pagos están relacionados?",
        "¿Qué puede causar timeouts del emisor o adquirente en el flujo de pagos?",
        "¿Cómo ayuda la idempotencia a evitar cobros duplicados?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state["pending_question"] = example
    st.divider()
    try:
        ready = client.ready()
        st.success(f"API: {ready['status']} · Neo4j: {ready['neo4j']}")
    except httpx.HTTPError:
        st.error("API/Neo4j no disponible")

st.title("GraphRAG para investigación de pagos")
st.markdown(
    "Consulta conocimiento operativo de pagos y expande automáticamente las entidades "
    "conectadas del grafo para aportar evidencia transaccional."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        payload = message.get("payload")
        if payload:
            with st.expander("Trazabilidad GraphRAG", expanded=False):
                trace = payload.get("trace", {})
                st.markdown(
                    f"""
                    <div class='trace-card'>
                    <b>Retriever:</b> {trace.get('retriever', '—')}<br/>
                    <b>Índice vectorial:</b> {trace.get('vector_index', '—')}<br/>
                    <b>Índice full-text:</b> {trace.get('fulltext_index', '—')}<br/>
                    <b>Expansión:</b> {trace.get('graph_expansion', '—')}<br/>
                    <b>Contextos:</b> {trace.get('returned_contexts', 0)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with st.expander("Contexto recuperado", expanded=False):
                render_contexts(payload.get("contexts", []))

pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Pregunta sobre rechazos, incidencias o controles de payment processing")
question = question or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recuperando contexto y expandiendo el grafo..."):
            try:
                payload = client.query(question, top_k)
                st.markdown(payload["answer"])
                with st.expander("Trazabilidad GraphRAG", expanded=True):
                    st.json(payload["trace"])
                with st.expander("Contexto recuperado", expanded=True):
                    render_contexts(payload.get("contexts", []))
                st.session_state.messages.append(
                    {"role": "assistant", "content": payload["answer"], "payload": payload}
                )
            except httpx.HTTPError as exc:
                st.error(f"Error consultando la API: {exc}")
