from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_utils import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MANIFEST_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_VECTORSTORE_DIR,
    TOPIC_ALL_LABEL,
    build_context_block,
    format_sources_markdown,
    load_index_manifest,
    resolve_default_embedding_model,
    resolve_default_llm_model,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
VECTORSTORE_DIR = PROJECT_ROOT / DEFAULT_VECTORSTORE_DIR
MANIFEST = load_index_manifest(VECTORSTORE_DIR)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or (
    MANIFEST or {}
).get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
DEFAULT_LLM_MODEL = resolve_default_llm_model()
DEFAULT_EMBED_MODEL = resolve_default_embedding_model(
    (MANIFEST or {}).get("embedding_model")
)

SYSTEM_PROMPT = """
Eres un asistente educativo especializado en transcripciones de Khan Academy.
Reglas obligatorias:
- Responde solo con el contexto recuperado.
- Si el contexto no basta o la respuesta no aparece de forma explícita, responde exactamente: "No encuentro esa información en los fragmentos recuperados."
- Nunca añadas explicaciones generales, ejemplos externos ni conocimiento propio después de indicar que falta contexto.
- No inventes datos ni uses conocimiento externo.
- Responde en español salvo que el usuario pida otro idioma.
- Cuando apoyes una idea en el contexto, cita [Fuente 1], [Fuente 2], etc.

Contexto recuperado:
{context}
""".strip()

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

CUSTOM_CSS = """
:root {
  --page-bg: linear-gradient(135deg, #f7efe0 0%, #edf4ea 100%);
  --panel-bg: rgba(255, 252, 246, 0.94);
  --card-bg: rgba(255, 255, 255, 0.86);
  --ink: #1f2d1f;
  --muted: #55645b;
  --accent: #2d6a4f;
  --accent-2: #c56b3a;
  --border: rgba(45, 106, 79, 0.18);
}

.gradio-container {
  background: var(--page-bg);
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
}

#hero-card,
#side-note,
.panel-shell {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: 20px;
  box-shadow: 0 18px 36px rgba(31, 45, 31, 0.08);
}

#hero-card {
  padding: 18px 22px;
  margin-bottom: 18px;
}

#hero-card h1 {
  margin: 0 0 8px 0;
  font-size: 2rem;
  line-height: 1.1;
}

#hero-card p {
  margin: 0;
  color: var(--muted);
}

#side-note {
  padding: 16px;
  margin-bottom: 14px;
}

.panel-shell {
  padding: 12px;
}

.gr-button-primary {
  background: linear-gradient(135deg, var(--accent), #4d8f6d) !important;
  border: none !important;
}

.gr-button-secondary {
  border: 1px solid var(--border) !important;
}
"""

INITIAL_SOURCES = "Las fuentes recuperadas aparecerán aquí después de la primera pregunta."

_VECTORSTORE: Chroma | None = None


def get_topics() -> list[str]:
    topics = [TOPIC_ALL_LABEL]
    if MANIFEST and MANIFEST.get("topics"):
        topics.extend(MANIFEST["topics"])
    return topics


def get_vectorstore() -> Chroma:
    global _VECTORSTORE

    if _VECTORSTORE is not None:
        return _VECTORSTORE

    if MANIFEST is None:
        raise RuntimeError(
            "No se encontró un índice vectorial local. Ejecuta `python index_data.py` antes de lanzar la app."
        )

    embeddings = OllamaEmbeddings(
        model=DEFAULT_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    _VECTORSTORE = Chroma(
        collection_name=MANIFEST.get("collection_name", DEFAULT_COLLECTION_NAME),
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
    )
    return _VECTORSTORE


def to_langchain_history(history: list[dict]) -> list[HumanMessage | AIMessage]:
    messages: list[HumanMessage | AIMessage] = []
    for item in history or []:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def retrieve_chunks(
    question: str,
    k_value: int,
    score_threshold: float,
    topic_filter: str,
) -> list[tuple]:
    vectorstore = get_vectorstore()
    metadata_filter = None
    if topic_filter and topic_filter != TOPIC_ALL_LABEL:
        metadata_filter = {"topic": topic_filter}

    raw_results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=int(k_value),
        filter=metadata_filter,
    )

    filtered_results = []
    for doc, score in raw_results:
        normalized_score = max(0.0, min(float(score), 1.0))
        if normalized_score >= float(score_threshold):
            filtered_results.append((doc, normalized_score))

    return filtered_results


def answer_question(
    question: str,
    history: list[dict],
    k_value: int,
    score_threshold: float,
    topic_filter: str,
    llm_model: str,
):
    history = list(history or [])
    clean_question = (question or "").strip()

    if not clean_question:
        return history, history, INITIAL_SOURCES, ""

    try:
        retrieved = retrieve_chunks(
            clean_question,
            k_value=k_value,
            score_threshold=score_threshold,
            topic_filter=topic_filter,
        )
    except Exception as exc:
        answer = (
            "No pude consultar el índice vectorial. Comprueba que el índice local "
            "existe y que Ollama está activo.\n\n"
            f"Detalle: {exc}"
        )
        updated_history = history + [
            {"role": "user", "content": clean_question},
            {"role": "assistant", "content": answer},
        ]
        return updated_history, updated_history, INITIAL_SOURCES, ""

    if not retrieved:
        answer = (
            "No encuentro contexto suficiente en las transcripciones seleccionadas "
            "para responder con seguridad. Prueba a reformular la pregunta, bajar "
            "el umbral o cambiar el filtro de tema."
        )
        sources_markdown = "No se recuperaron fragmentos con el umbral actual."
    else:
        model_name = (llm_model or DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL
        chain = PROMPT | ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        ) | StrOutputParser()
        context_block = build_context_block(retrieved)

        try:
            answer = chain.invoke(
                {
                    "context": context_block,
                    "history": to_langchain_history(history),
                    "question": clean_question,
                }
            )
        except Exception as exc:
            answer = (
                f"No pude generar la respuesta con el modelo `{model_name}`. "
                "Verifica que el modelo exista en Ollama.\n\n"
                f"Detalle: {exc}"
            )

        lower_answer = answer.lower()
        insufficient_markers = [
            "no encuentro",
            "no hay suficiente informacion",
            "no dispongo de informacion",
            "el contexto no contiene",
            "the context does not contain",
        ]
        if any(marker in lower_answer for marker in insufficient_markers):
            answer = "No encuentro esa información en los fragmentos recuperados."

        sources_markdown = format_sources_markdown(retrieved)

    updated_history = history + [
        {"role": "user", "content": clean_question},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, updated_history, sources_markdown, ""


def clear_chat():
    return [], [], INITIAL_SOURCES, ""


def build_status_markdown() -> str:
    if MANIFEST is None:
        return (
            "### Estado del proyecto\n"
            "- Índice vectorial: pendiente\n"
            "- Ejecuta `python index_data.py` para crear el índice.\n"
            f"- Modelo LLM por defecto: `{DEFAULT_LLM_MODEL}`\n"
            f"- Modelo de embeddings esperado: `{DEFAULT_EMBED_MODEL}`"
        )

    return (
        "### Estado del proyecto\n"
        f"- Chunk count: **{MANIFEST.get('chunk_count', 0)}**\n"
        f"- Embeddings: `{MANIFEST.get('embedding_model', DEFAULT_EMBED_MODEL)}`\n"
        f"- Ollama base URL: `{OLLAMA_BASE_URL}`\n"
        f"- Manifest: `{VECTORSTORE_DIR / DEFAULT_MANIFEST_NAME}`"
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Khan Academy RAG con Ollama",
        fill_width=True,
    ) as demo:
        gr.Markdown(
            """
<div id="hero-card">
  <h1>Khan Academy RAG Studio</h1>
  <p>Consulta transcripciones educativas con Ollama, LangChain y una memoria conversacional simple desde una interfaz local en Gradio.</p>
</div>
""".strip()
        )

        with gr.Row():
            with gr.Column(scale=5, elem_classes="panel-shell"):
                chatbot = gr.Chatbot(
                    label="Conversación",
                    height=560,
                    layout="bubble",
                    placeholder="Haz una pregunta sobre las transcripciones del dataset limpio.",
                )
                history_state = gr.State([])
                question_box = gr.Textbox(
                    label="Pregunta",
                    placeholder="Ejemplo: What are the inputs of photosynthesis?",
                )
                with gr.Row():
                    send_button = gr.Button("Preguntar", variant="primary")
                    clear_button = gr.Button("Limpiar historial", variant="secondary")

            with gr.Column(scale=3):
                gr.Markdown(build_status_markdown(), elem_id="side-note")
                with gr.Accordion("Ajustes del retriever", open=True):
                    k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        label="Número de fragmentos (k)",
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.05,
                        label="Umbral mínimo de similitud",
                    )
                    topic_dropdown = gr.Dropdown(
                        choices=get_topics(),
                        value=TOPIC_ALL_LABEL,
                        label="Filtro por tema",
                    )
                    llm_model_box = gr.Textbox(
                        value=DEFAULT_LLM_MODEL,
                        label="Modelo LLM de Ollama",
                    )

                with gr.Accordion("Fuentes recuperadas", open=True):
                    sources_markdown = gr.Markdown(INITIAL_SOURCES)

        inputs = [
            question_box,
            history_state,
            k_slider,
            threshold_slider,
            topic_dropdown,
            llm_model_box,
        ]
        outputs = [chatbot, history_state, sources_markdown, question_box]

        question_box.submit(answer_question, inputs=inputs, outputs=outputs)
        send_button.click(answer_question, inputs=inputs, outputs=outputs)
        clear_button.click(
            clear_chat,
            inputs=None,
            outputs=[chatbot, history_state, sources_markdown, question_box],
        )

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(css=CUSTOM_CSS)
