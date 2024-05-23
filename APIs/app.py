# API
from fastapi import FastAPI
import uvicorn

from langserve import add_routes

# Chatbot
from langchain.prompts import ChatPromptTemplate
## LLM (LLama2 and Gemma)
from langchain_community.llms import Ollama

# Environment
import os
from dotenv import load_dotenv

# Retrieve the environment variables
load_dotenv()
# Storing of the results of the inferences
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# API server for Langchain
app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    decsription = "A simple API Server for Langchain"
)

# Llama2
Llama_LLM = Ollama(model="llama2")
# Gemma
Gemma_LLM = Ollama(model="gemma")

# Prompt template
prompt_Llama = ChatPromptTemplate.from_template("Llama write me an essay about {topic} with 100 words in english")
prompt_Gemma = ChatPromptTemplate.from_template("Gemma write me an essay about {topic} with 100 words in english")

add_routes(
    app,
    prompt_Llama|Llama_LLM,
    path = "/Llama2",
)

add_routes(
    app,
    prompt_Gemma|Gemma_LLM,
    path = "/Gemma",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)