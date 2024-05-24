import requests
import streamlit as st

def get_Llama2_response(input_text):

    response = requests.post("http://localhost:8000/Llama2/invoke",
                             json={'input':{'topic':input_text}}
                    )

    return response.json()['output']

def get_Gemma_response(input_text):

    response = requests.post("http://localhost:8000/Gemma/invoke",
                             json={'input':{'topic':input_text}}
                    )

    return response.json()['output']

st.title('Langchain Demo With Multiple LLMs')
input_text_Llama = st.text_input("Llama please write an essay on")
input_text_Gemma = st.text_input("Gemma please write an essay on")

if input_text_Llama:
    st.write(get_Llama2_response(input_text_Llama))
if input_text_Gemma:
    st.write(get_Gemma_response(input_text_Gemma))

# To make this client work we need to start the API first with app.py

