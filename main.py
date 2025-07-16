from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

Gemini_api_key= os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key= Gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model= OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

class quiz(BaseModel):
    question: str
    option: list[str]
    answer: str
    
agent = Agent(
    name="Quiz Assistant",
    instructions="You create a quiz in specific topic the user give to you",
    output_type=quiz
)
 
print("\n I'm a Quiz making-assistant, How can i help you \n ")
prompt = input("Enter your prompt: ")

result = Runner.run_sync(
    agent,
    prompt,
    run_config=config
)

print(result.final_output , "\n" )