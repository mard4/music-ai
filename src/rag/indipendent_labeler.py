from datapizza.agents.agent import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
from datapizza.modules.prompt import ChatPromptTemplate
import pandas as pd
import os
from datapizza.type import Media, MediaBlock, TextBlock


client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL", "gpt-4.1"))

@tool
def get_weather(city: str) -> str:
    return f""" it's sunny all the week in {city}"""

def read_txt_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()
    
weather_agent = Agent(
    name="clean_label",
    client=client,
    system_prompt=read_txt_file("src/rag/prompts/clean_label.txt"),
    #tools=[get_weather]
)

web_search_agent = Agent(
    name="orchestrator",
    client=client,
    system_prompt = ChatPromptTemplate(
        user_prompt_template=read_txt_file("src/rag/prompts/orchestrator.txt"),
        retrieval_prompt_template=read_txt_file("src/rag/prompts/orchestrator.txt"),
    )
    #tools=[DuckDuckGoSearchTool()]
)

planner_agent = Agent(
    name="generator",
    client=client,
    system_prompt=read_txt_file("src/rag/prompts/generator.txt")
)

planner_agent.can_call([weather_agent, web_search_agent])

# usage

media = Media(
    media_type="audio",      # Specifica il tipo "audio"
    source_type="path",      # La fonte è un percorso file
    source="sample.mp3",     # Il percorso del tuo file audio
    extension="mp3"          # L'estensione del file
)

response = planner_agent.run(
    "How can i make this sound?"
)

print(response.text)
