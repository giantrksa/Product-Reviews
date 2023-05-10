import os
import openai

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import requests
from bs4 import BeautifulSoup
import langchain as lc
import pinecone
from sentence_transformers import SentenceTransformer
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from llama_index import Document, ServiceContext

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from llama_index import LLMPredictor


import streamlit as st

def app():
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    os.environ["OPENAI_API_KEY"] = "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"  
    openai.api_key = "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"

    directory = '/workspace/JupyterLab (DATA)/NLP_GPT/development/data'

    embeddings = HuggingFaceEmbeddings()


    pinecone.init(
        api_key="5deaf73e-7868-4138-a662-72738d095818",  # find at app.pinecone.io
        environment="northamerica-northeast1-gcp"  # next to api key in console
        )

    index_name = "langchain-demo"

    index = Pinecone.from_existing_index(index_name, embeddings)

    def get_similiar_docs(query,k=3,score=False):
        if score:
            similar_docs = index.similarity_search_with_score(query,k=k)
        else:
            similar_docs = index.similarity_search(query,k=k)
        return similar_docs

    llm = OpenAI(model_name="text-davinci-003")
    chain = load_qa_chain(llm, chain_type="stuff")

    def get_answer(query):
        similar_docs = get_similiar_docs(query)
        answer =  chain.run(input_documents=similar_docs, question=query)
        return  answer

    st.title('Product Specification Optimizer')

    # Input text for theme of content
    theme = st.text_input("Ask me about product that you interest:")

    # Generate button
    if st.button("Generate"):
        if theme:
            st.write("Chat bot:")
            generated_content = get_answer(theme)
            st.write(generated_content)
        else:
            st.warning("Please enter a prompt to generate information and review.")
