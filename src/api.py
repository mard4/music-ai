import shutil
import sys
import os
import uuid
from typing import Optional, List

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In produzione metti ["http://localhost:4200"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("public_audio", exist_ok=True)
app.mount("/public", StaticFiles(directory="public_audio"), name="public")

# Modello dati per la richiesta
class ChatRequest(BaseModel):
    query: str
    files: Optional[list] = []


class ChatResponse(BaseModel):
    answer: str
    context_used: List[dict] = []
    suggestions: List[str] = []

@app.get("/")
async def root():
    return {"status": "online", "message": "AI Sound Assistant is running"}

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"status": "online", "message": "AI Sound Assistant is running"}


@app.post("/ask", response_model=ChatResponse)
async def chat_endpoint(
        query: str = Form(""),
        files: list[UploadFile] = File(default=[])  # Usa list[UploadFile]
):
    temp_file_path = None
    try:
        logger.info(f"API Request ricevuta. Query: {query}, Files: {len(files)}")

        # 1. GESTIONE SALVATAGGIO FILE SU DISCO
        if files:
            # Prendiamo il primo file (per semplicità)
            uploaded_file = files[0]

            # Crea directory temporanea se non esiste
            temp_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)

            temp_file_path = os.path.join(temp_dir, uploaded_file.filename)

            logger.info(f"Salvataggio file temporaneo in: {temp_file_path}")

            # Scrittura dei byte su disco
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

        # 2. ESECUZIONE WORKFLOW
        wf = app.state.workflow

        # Passiamo sia la query testuale che il path del file (se esiste)
        final_response = await wf.run(user_input=query, file_path=temp_file_path)

        return ChatResponse(
            answer=str(final_response),
            context_used=[],
            suggestions=[]
        )

    except Exception as e:
        logger.error(f"Errore API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 3. PULIZIA (Opzionale ma consigliata)
        # Rimuovi il file temporaneo dopo aver generato la risposta per non intasare il server
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"File temporaneo rimosso: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Impossibile rimuovere file temporaneo: {cleanup_error}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)