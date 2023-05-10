# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from apps import content_generator


# Define the home page
def home_page():
    # Libraries
    from PIL import Image

    # Title
    st.title('Product Specification Optimizer')

    st.subheader('Given')
    st.write(
        """
        
        ChatGPT or GPT-3 was trained using the public web text data up to 2021.
        Then: As a result, it doesn’t know the information on 2023.

        """
    )

    st.subheader('Story')
    st.write(
        """
        As an X company, I’d like to have a question-and-answer app powered by an AI model,
        My product is Y and it was released in 2023,
        I want the app can answer any product specification using the prompt.
        """
    )

    st.subheader('Requirements')
    st.write(
        """
        - Y product is a product. You are free to use any product, based on your
        interest. (e.g iPhone 14, Samsung S22, Wuling Alvez, etc).
        - The external data about the product is up to you. You can use product
        specification PDF, Youtube transcript, Google search, etc.
        - Use Python.
        - Use Langchain https://python.langchain.com/en/latest/index.html
        - Use any foundation model / LLMs you’d like, ChatGPT, HuggingFace models,
        your own custom model, etc.
        - You can use Jupyter Notebook or a Python file that is executed manually.
        - Any implementation of Langchain modules is a plus (e.g prompt templates,
        indexes, chains, agents, memory)
        - Vector-based DB implementation is a plus (e.g pinecone)
        - Any visualization is a plus
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info('**Data Scientist: Gian**')
    with c2:
        st.info('**Conceptor: Team Cookies**')
    with c3:
        st.info('**GitHub: [@giantrksa](https://github.com/giantrksa/)**')


# Define the pages of our app
PAGES = {
    "Introduction":home_page,
    "Chat Reviews Here": content_generator.app,
}

# Define the sidebar menu
with st.sidebar:
    st.sidebar.image("seo.png", use_column_width=True, output_format="PNG")
    st.sidebar.title('Product Specification Optimizer')
    page = option_menu("Features", list(PAGES.keys()))

# Display the selected page
PAGES[page]()
