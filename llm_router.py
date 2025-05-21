# llm_router.py
import os
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import google.generativeai as genai

router = APIRouter()

# ─── Models & Schemas ──────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    max_results: int = 50

class QuestionResponse(BaseModel):
    answer: str
    sources: List[dict]

# ─── Lazy Gemini Setup ─────────────────────────────────────────────────────────

_genai_configured = False
def _ensure_gemini():
    global _genai_configured
    if not _genai_configured:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        _genai_configured = True

def _get_model():
    _ensure_gemini()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.types.GenerationConfig(temperature=0.5)
    )

# ─── Route ────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "A pergunta não pode estar vazia")

    print(f"[LLM] Pergunta: '{question}' (max={request.max_results})")

    # ─── Lazy-import Pinecone search ──────────────────────────────────────────
    try:
        from pinecone_utils import buscar_documentos
    except ImportError as e:
        raise HTTPException(500, f"Erro interno: {e}")

    try:
        documentos = buscar_documentos(question, request.max_results)
    except Exception as e:
        print(f"[LLM] Erro na busca semântica: {e}")
        raise HTTPException(500, "Erro ao buscar documentos.")

    if not documentos:
        raise HTTPException(404, "Nenhum documento relevante encontrado.")

    # ─── Build prompt ─────────────────────────────────────────────────────────
    context = "\n\n".join(
        f"[Doc {i+1} - {doc['arquivo']}]\n{doc['texto']}"
        for i, doc in enumerate(documentos)
    )

    prompt = (
        "Você é um assistente especializado em contratos imobiliários.\n"
        "Responda de forma:\n"
        "1. DETALHADA\n2. ESPECÍFICA\n"
        "3. ESTRUTURADA\n4. BASEADA EM EVIDÊNCIAS\n\n"
        f"Documentos:\n{context}\n\nPergunta: {question}"
    )

    # ─── Call Gemini ──────────────────────────────────────────────────────────
    try:
        model = _get_model()
        chat = model.start_chat()
        response = chat.send_message(prompt)
        answer = response.text
    except Exception as e:
        print(f"[LLM] Erro ao gerar resposta: {e}")
        raise HTTPException(500, f"Erro ao gerar resposta. {e}")

    elapsed = time.time() - start_time
    print(f"[LLM] Resposta em {elapsed:.2f}s.")

    return QuestionResponse(
        answer=answer,
        sources=[{"filename": d["arquivo"], "text": d["texto"]} for d in documentos]
    )
