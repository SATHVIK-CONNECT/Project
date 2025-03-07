from groq import Groq
import base64
import os
import streamlit as st
import json
import tempfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from streamlit_lottie import st_lottie

# Layout
st.set_page_config(page_title="Spark AI", page_icon="⚡")

# Set up Groq API Key
os.environ['GROQ_API_KEY'] = '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'

# Load the lottie animation for loading
def load_lottie():
    with open("pdfload.json", 'r', encoding="UTF-8") as f:
        return json.load(f)

pdfload = load_lottie()

# Initialize Groq API
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def create_vector_db_out_of_the_uploaded_pdf_file(pdf_file):
    # Clear previous session state if it exists
    if "vector_store" in st.session_state:
        del st.session_state["vector_store"]
        del st.session_state["embeddings"]
        del st.session_state["loader"]
        del st.session_state["text_document_from_pdf"]
        del st.session_state["text_splitter"]
        del st.session_state["final_document_chunks"]

    # Create temporary file for the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        pdf_file_path = temp_file.name

    # Create a placeholder for the loading animation
    loading_placeholder = st.empty()
    st_lottie(pdfload, width=150, height=150)

    # Load embeddings and process PDF
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    st.session_state.loader = PyPDFLoader(pdf_file_path)
    st.session_state.text_document_from_pdf = st.session_state.loader.load()

    # Split document into chunks
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)

    # Create vector store from the chunks
    st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)

    # Remove the loading animation once processing is done
    loading_placeholder.empty()

# Streamlit file uploader
pdf_input_from_user = st.file_uploader("Upload the PDF file", type=['pdf'])

if pdf_input_from_user is not None:
    if st.button("Create the Vector DB from the uploaded PDF file"):
        create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
        st.success("Vector Store DB for this PDF file is Ready")

if "vector_store" in st.session_state:
    user_prompt = st.text_input("Enter Your Question related to the uploaded PDF")

    if st.button('Submit Prompt'):
        with st.spinner('Generating output...'):
            if user_prompt:
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                response = retrieval_chain.invoke({'input': user_prompt})
                st.write(response['answer'])
            else:
                st.error('Please write your prompt')
