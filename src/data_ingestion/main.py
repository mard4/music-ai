from data_ingestion.ingestors.ingest_audio import ingest_audio_catalog
from data_ingestion.ingestors.ingest_parameters import ingest_socialfx_vectors
import asyncio

async def main_ingestor():
    await ingest_audio_catalog()
    await ingest_socialfx_vectors()

if __name__ == "__main__":
    asyncio.run(main_ingestor())
