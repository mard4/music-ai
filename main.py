import asyncio

from data_retrievial.sample_focus.main import main_samplefocus
from data_retrievial.socialfx.main import main_socialfx
from data_ingestion.main import main_ingestor
from rag.main import main_workflow

if __name__ == "__main__":
    asyncio.run(main_samplefocus())
    asyncio.run(main_socialfx())
    asyncio.run(main_ingestor())

    asyncio.run(main_workflow())
