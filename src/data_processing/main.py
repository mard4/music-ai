import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.audio_processor import samplefocus_fix

if __name__ == "__main__":
    asyncio.run(samplefocus_fix())