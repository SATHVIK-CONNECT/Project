import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Placeholder function for answering queries (replace with actual logic)
def get_answer_from_custom_service(relevant_text, user_query):
    """
    This function will handle the custom logic for generating an answer based on the relevant text and user query.
    Replace it with actual integration logic.
    """
    # Placeholder answer based on the relevant text and user query.
    answer = "This is a placeholder answer. Replace this with actual response generation logic."
    return answer

# Function to create the vector database from the uploaded PDF
def create_vector_db_out_of_the_uploaded_pdf_file(pdf_file):
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        pdf_file_path = temp_file.name

    # Use HuggingFace embeddings to encode the text into vectors
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    
    # Load the PDF document
    loader = PyPDFLoader(pdf_file_path)
    text_document_from_pdf = loader.load()

    # Split the document into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_document_chunks = text_splitter.split_documents(text_document_from_pdf)

    # Create a FAISS vector store from the document chunks
    vector_store = FAISS.from_documents(final_document_chunks, embeddings)
    
    return vector_store

# Streamlit app UI
pdf_input_from_user = st.file_uploader("Upload the PDF file", type=['pdf'])

if pdf_input_from_user is not None:
    # Check if the vector store already exists in session state
    if "vector_store" not in st.session_state:
        # Create the vector store and store it in session state for later use
        vector_store = create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
        st.session_state.vector_store = vector_store
        st.session_state.pdf_uploaded = True
    else:
        vector_store = st.session_state.vector_store
        st.session_state.pdf_uploaded = True

    user_prompt = st.text_input("Enter Your Question related to the uploaded PDF")

    if st.button('Submit Prompt'):
        with st.spinner('Generating output...'):
            if user_prompt:
                # Retrieve the relevant document chunks based on the user prompt
                relevant_chunks = vector_store.similarity_search(user_prompt, k=3)  # 'k=3' for top 3 most relevant chunks

                # Combine the relevant chunks' content into one string
                relevant_text = " ".join([chunk.page_content for chunk in relevant_chunks])

                # Use the custom service to process the relevant chunks and answer the query
                answer = get_answer_from_custom_service(relevant_text, user_prompt)

                # Display the answer to the user
                st.write(answer)
            else:
                st.error('Please write your prompt')
else:
    if "pdf_uploaded" in st.session_state and st.session_state.pdf_uploaded:
        st.write("You have already uploaded a PDF. Please enter your question.")
    else:
        st.write("Please upload a PDF file first")
