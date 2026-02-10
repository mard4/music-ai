import asyncio

from data_ingestion.ingestors.enrich_audio_doublevectors import EnrichedCollectionIngestor
from data_ingestion.ingestors.ingest_parameters import SocialFXIngestor


# async def main_ingestor():
#
#     audio_ingestor = AudioCatalogIngestor()
#     await audio_ingestor.run()
#
#     params_ingestor = SocialFXIngestor()
#     await params_ingestor.run()
#
#
# if __name__ == "__main__":
#     asyncio.run(main_ingestor())

if __name__ == "__main__":
    asyncio.run(EnrichedCollectionIngestor().run())