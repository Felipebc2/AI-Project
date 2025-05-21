import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import uvicorn
from pinecone import Pinecone
import google.generativeai as genai
from llm_router import router as llm
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "brito-ai")

# Configuração da API do Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Inicializa o modelo de embedding do Gemini
embedding_model = genai.GenerativeModel(model_name="models/embedding-001")

# Variáveis globais para conexão com Pinecone
pc = None
index = None
connection_attempts = 0
MAX_RECONNECT_ATTEMPTS = 3

def conectar_pinecone():
    """Função para conectar ao Pinecone com retry automático"""
    global pc, index, connection_attempts
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
        stats = index.describe_index_stats()
        print(f"Conexão com o índice '{PINECONE_INDEX_NAME}' estabelecida com sucesso!")
        print(f"Total de vetores no índice: {stats.get('total_vector_count', 0)}")
        connection_attempts = 0
        return True
    except Exception as e:
        connection_attempts += 1
        print(f"Erro ao inicializar Pinecone (tentativa {connection_attempts}/{MAX_RECONNECT_ATTEMPTS}): {e}")
        if connection_attempts < MAX_RECONNECT_ATTEMPTS:
            import time
            print(f"Tentando reconectar em 2 segundos...")
            time.sleep(2)
            return conectar_pinecone()
        else:
            print("Número máximo de tentativas excedido. Falha na conexão com Pinecone.")
            return False

conexao_bem_sucedida = conectar_pinecone()

def gerar_embedding(texto):
    """
    Gera um embedding usando o modelo do Gemini.

    Args:
        texto: Texto para gerar o embedding

    Returns:
        Lista com o embedding
    """
    try:
        embedding_response = genai.embed_content(
            model="models/embedding-001",
            content=texto,
            task_type="retrieval_document" # Good practice for document retrieval
        )
        return embedding_response["embedding"]
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        raise

app = FastAPI(title="Contratus AI API", 
              description="API para consulta semântica de contratos usando Pinecone",
              version="2.0.0",
              debug=True
              )

app.include_router(llm,prefix="/llm")

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class ContratoBase(BaseModel):
    arquivo: str
    texto: str

class ContratoResponse(ContratoBase):
    score: float = 0.0

class SearchResponse(BaseModel):
    resultados: List[ContratoResponse]
    total: int

@app.get("/")
def read_root():
    global index
    try:
        if index:
            stats = index.describe_index_stats()
            return {
                "status": "online", 
                "message": "Contratus AI API está funcionando com Pinecone!",
                "pinecone_status": "conectado",
                "total_vetores": stats.get("total_vector_count", 0)
            }
        else:
            if conectar_pinecone():
                return {
                    "status": "online", 
                    "message": "Contratus AI API está funcionando com Pinecone!",
                    "pinecone_status": "reconectado"
                }
            else:
                return {
                    "status": "parcial", 
                    "message": "API está online, mas sem conexão com Pinecone.",
                    "pinecone_status": "desconectado"
                }
    except Exception as e:
        conectar_pinecone()
        return {
            "status": "degradado", 
            "message": "API está online, mas com problemas de conexão ao Pinecone.",
            "error": str(e)
        }

@app.get("/contratos", response_model=SearchResponse)
def listar_contratos(
    skip: int = Query(0, description="Número de registros para pular"),
    limit: int = Query(10, description="Número máximo de registros para retornar")
):
    global index
    if not index and not conectar_pinecone():
        raise HTTPException(
            status_code=503, 
            detail="Serviço temporariamente indisponível. Não foi possível conectar ao Pinecone."
        )
    try:
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        dummy_vector = [0.0] * 768  # Ajuste a dimensão conforme o modelo de embedding utilizado
        resultados_query = index.query(
            vector=dummy_vector,
            top_k=skip + limit,
            include_metadata=True
        )
        matches = resultados_query.matches[skip:skip+limit] if resultados_query.matches else []
        resultados = []
        for match in matches:
            resultados.append(ContratoResponse(
                arquivo=match.metadata.get("arquivo", ""),
                texto=match.metadata.get("texto", ""),
                score=match.score
            ))
        return SearchResponse(resultados=resultados, total=total)
    except Exception as e:
        print(f"Erro ao listar contratos: {e}")
        if conectar_pinecone():
            try:
                return listar_contratos(skip=skip, limit=limit)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Erro ao listar contratos: {str(e)}")

@app.get("/contratos/busca", response_model=SearchResponse)
def buscar_contratos(
    q: str = Query(..., description="Consulta para busca"),
    limit: int = Query(5, description="Número máximo de resultados")
):
    global index
    if not q:
        raise HTTPException(status_code=400, detail="A consulta não pode estar vazia")
    if not index and not conectar_pinecone():
        raise HTTPException(
            status_code=503, 
            detail="Serviço temporariamente indisponível. Não foi possível conectar ao Pinecone."
        )
    try:
        query_embedding = gerar_embedding(q)
        resultados_query = index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True
        )
        resultados = []
        for match in resultados_query.matches:
            resultados.append(ContratoResponse(
                arquivo=match.metadata.get("arquivo", ""),
                texto=match.metadata.get("texto", ""),
                score=match.score
            ))
        total = len(resultados)
        return SearchResponse(resultados=resultados, total=total)
    except Exception as e:
        print(f"Erro na busca: {e}")
        if conectar_pinecone():
            try:
                return buscar_contratos(q=q, limit=limit)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Erro ao realizar a busca: {str(e)}")

@app.get("/contratos/arquivos")
def listar_arquivos():
    global index
    if not index and not conectar_pinecone():
        raise HTTPException(
            status_code=503, 
            detail="Serviço temporariamente indisponível. Não foi possível conectar ao Pinecone."
        )
    try:
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        dummy_vector = [0.0] * 768  # Ajuste a dimensão conforme o modelo de embedding utilizado
        resultados_query = index.query(
            vector=dummy_vector,
            top_k=min(total, 1000),
            include_metadata=True
        )
        arquivos = set()
        for match in resultados_query.matches:
            arquivo = match.metadata.get("arquivo")
            if arquivo:
                arquivos.add(arquivo)
        return {"arquivos": list(arquivos)}
    except Exception as e:
        print(f"Erro ao listar arquivos: {e}")
        if conectar_pinecone():
            try:
                return listar_arquivos()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Erro ao listar arquivos: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api_pinecone:app", host="127.0.0.1", port=8000, reload=True)
