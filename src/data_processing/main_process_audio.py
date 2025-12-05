import asyncio
from ProcessAudio import create_audio_processor
from config.settings import mongo_config


async def process_audio_files() -> None:
    """Main function to process audio files."""

    processor = await create_audio_processor(mongo_config=mongo_config)

    await processor.process_audio_files()


if __name__ == "__main__":

    asyncio.run(process_audio_files())