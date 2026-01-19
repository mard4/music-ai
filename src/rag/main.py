import asyncio
import os
import sys
from pathlib import Path
from rag.workflow import Workflow

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main_workflow():
    wf = Workflow()

    res = await wf.run("Find me a bright bass")
    print(res)

if __name__ == "__main__":

    asyncio.run(main_workflow())
