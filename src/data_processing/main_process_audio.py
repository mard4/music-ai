import asyncio
import logging

logging.getLogger('pymongo').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('librosa').setLevel(logging.WARNING)

from config.settings import settings
from ProcessAudio import create_audio_processor

async def process_audio_files() -> None:
    """Main function to process audio files."""

    processor = await create_audio_processor()

    await processor.process_audio_files()


if __name__ == "__main__":

    asyncio.run(process_audio_files())