import streamlit as st
import os
import time

# LangChain for Groq
from langchain_groq import ChatGroq
# Read from website
from langchain_community.document_loaders import WebBaseLoader
# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
# Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Prompt
from langchain_core.prompts import ChatPromptTemplate
# Vector store
from langchain_community.vectorstores import FAISS
# Stuff document chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Retieval chain
from langchain.chains import create_retrieval_chain

# Retrieve the environment variables
from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ["GROQ_API_KEY"]

# HF embedding
langchain_embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if "vector" not in st.session_state:
    st.session_state.embeddings = langchain_embed_model
    st.session_state.loader = WebBaseLoader(web_path="https://en.wikipedia.org/wiki/Taylor_Swift")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:100])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# Model
st.title("ChatGroq Demo with mixtral model")
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="mixtral-8x7b-32768"
        )

# Prompt
prompt = ChatPromptTemplate.from_template(
"""
You are the official biographer of Taylor Swift. 
Don't rush your answer and work out your solution to the user query cleverly.
Think step by step before providing a detailed and accurate answer.
Answer the questions based on the provided context only.
Please provide the most accurate and not harmful response based on the question of the user.
<context>
{context}
<context>
Questions:{input}
"""
)

# Document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Retriever
retriever = st.session_state.vectors.as_retriever()

# Retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Prompt to insert")

if prompt:
    # Get the processing time of the user query
    start_time = time.process_time()
    # Get the response of the user query
    response = retrieval_chain.invoke({"input":prompt})

    print("Response time of the query:", time.process_time() - start_time)

    st.write(response['answer']) 

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant parts of the document for the query answer
        for i, document in enumerate(response["context"]):

            st.write(document.page_content)
            st.write("--------------------------------")










