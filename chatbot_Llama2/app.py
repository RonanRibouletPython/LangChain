# Aim of the first project = chatbot application
# We will try to create a chatbot with a open source LLM
# We will see features of LangSmith, Chains and Agents and Model I/O

# Use Ollama open source LLMs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Building web apps with Streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# Retrieve the environment variables
load_dotenv()
# Storing of the results of the inferences
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Designing the prompt template

# The prompt template is a list of messages
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You work like a sentiment analysis tool. You will be able to understand the mood of sentences that a user writes. Your answer will be a sentiment score between -1 and 1 with 0 being neutral and -1 being negative and 1 being positive"),
        ("user", "Sentence:{sentence}")
    ]
)

# Streamlit framework
st.title('Langchain Demo With Llama 2')
input_text = st.text_input("Search the topic u want")

# Use the OpenAI API
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'sentence':input_text}))