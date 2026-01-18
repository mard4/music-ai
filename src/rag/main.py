import asyncio
import os
import sys
from pathlib import Path
from rag.workflow import Workflow

sys.path.insert(0, str(Path(__file__).parent.parent))


if __name__ == "__main__":

    async def main():
        wf = Workflow()

        res = await wf.run("Find me a bright acid bass")
        print(res)


    asyncio.run(main())
