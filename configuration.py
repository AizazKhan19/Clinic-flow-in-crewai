from crewai import LLM
import dotenv
import os

dotenv.load_dotenv()

# llm configuration

llm = LLM(
    model = "gpt-4o-mini",
    api_key= os.getenv("OPENAI_API_KEY"),
    temperature= 0.5
)