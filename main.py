import asyncio

from data_ingestion.main import main_ingestor
from data_retrievial.sample_focus.main import main_samplefocus
from data_retrievial.socialfx.main import main_socialfx

if __name__ == "__main__":
    asyncio.run(main_samplefocus())
    asyncio.run(main_socialfx())

    asyncio.run(main_ingestor())

    ##        res = asyncio.run(orchestrator.process_request(f"Analizza il {path_test}"))
