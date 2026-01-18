import asyncio
from config.settings import mongo_config
from data_retrievial.socialfx.socialfx_extractor import create_socialfx_extractor

if __name__ == "__main__":

    async def main_socialfx():

        kb_extractor = await create_socialfx_extractor(mongo_config)
        await kb_extractor.save_kb_to_mongo()


    asyncio.run(main_socialfx())