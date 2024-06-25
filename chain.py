import streamlit as st
import tempfile
import os
from langchain.chroma import Chroma
from langchain.openai import OpenAIEmbeddings
from langchain.loader import PyPDFLoader
from langchain.chain import load_summarize_chain
from langchain.openai import ChatOpenAI

# Initialize session state variables
if 'openai_api_key' not in st.session_state:
	st.session_state.openai_api_key = ""

if 'serper_api_key' not in st.session_state:
	st.session_state.serper_api_key = ""

st.set_page_config(page_title="Home", page_icon="🦜️🔗")

st.header("Welcome to LangChain! 👋")

# Get OpenAI API key and source document input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
source_doc = st.file_uploader("Source Document", label_visibility="collapsed", type="pdf")

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner('Please wait...'):
              # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
              with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                  tmp_file.write(source_doc.read())
              loader = PyPDFLoader(tmp_file.name)
              pages = loader.load_and_split()
              os.remove(tmp_file.name)

              # Create embeddings for the pages and insert into Chroma database
              embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
              vectordb = Chroma.from_documents(pages, embeddings)

              # Initialize the OpenAI module, load and run the summarize chain
              llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
              chain = load_summarize_chain(llm, chain_type="stuff")
              search = vectordb.similarity_search(" ")
              summary = chain.run(input_documents=search, question="Write a summary within 200 words.")

              st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")