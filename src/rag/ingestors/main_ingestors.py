from rag.ingestors.ingest_audio import ingest_audio_catalog
from rag.ingestors.ingest_parameters import ingest_socialfx_vectors
import asyncio

if __name__ == "__main__":
    asyncio.run(ingest_audio_catalog())
    asyncio.run(ingest_socialfx_vectors())
