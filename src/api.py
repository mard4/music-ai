import sys
import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__)))
from rag.workflow import Workflow
from rag.utils import logger


# --- CONFIGURAZIONE LIFESPAN ---
# Questo serve a caricare il Workflow (e i modelli pesanti) UNA VOLTA SOLA all'avvio
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- AVVIO SERVER API ---")
    logger.info("Caricamento modelli e workflow in memoria...")

    # Inizializziamo il workflow e lo salviamo nello state dell'app
    app.state.workflow = Workflow()

    logger.info("Server pronto a ricevere richieste.")
    yield
    logger.info("--- SPEGNIMENTO SERVER ---")


# --- APP SETUP ---
app = FastAPI(title="AI Sound Assistant API", lifespan=lifespan)


# Modello dati per la richiesta
class ChatRequest(BaseModel):
    query: str



# Modello dati per la risposta
class ChatResponse(BaseModel):
    response: str
    # Se volessi debuggare anche i dati grezzi, potresti aggiungere un campo qui


# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "online", "message": "AI Sound Assistant is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principale per interagire con l'AI.
    """
    try:
        logger.info(f"API Request ricevuta: {request.query}")

        wf: Workflow = app.state.workflow
        final_response = await wf.run(request.query)

        return ChatResponse(response=final_response)

    except Exception as e:
        logger.error(f"Errore API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- ENTRY POINT ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)