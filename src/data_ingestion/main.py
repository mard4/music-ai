import asyncio
from data_ingestion.ingestors.ingest_audio import AudioCatalogIngestor
from data_ingestion.ingestors.ingest_parameters import SocialFXIngestor


async def main():

    audio_ingestor = AudioCatalogIngestor()
    await audio_ingestor.run()

    params_ingestor = SocialFXIngestor()
    await params_ingestor.run()


if __name__ == "__main__":
    asyncio.run(main())