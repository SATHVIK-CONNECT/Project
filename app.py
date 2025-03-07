from groq import Groq
import base64
import os
import streamlit as st

import json
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from streamlit_lottie import  st_lottie
import tempfile

# Layout
st.set_page_config(page_title="Spark AI", page_icon="⚡")

# Set up Groq API Key
os.environ['GROQ_API_KEY'] = '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'

# Styling
canvas = st.markdown("""
    <style>
        header{ visibility: hidden; }   
    </style> """, unsafe_allow_html=True)


# Function to generate caption
def generate(uploaded_image, prompt):
    base64_image = base64.b64encode(uploaded_image.read()).decode('utf-8')
    client = Groq()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{base64_image}',
                        },
                    },
                ],
            }
        ],
        model='llama-3.2-90b-vision-preview',
    )
    return chat_completion.choices[0].message.content

st.title("⚡Spark AI")

tab_titles = [
    "Home",
    "Vision Instruct",
    "File Query",
    "About",
]

vision = [
    "The vision instruct system utilizes visual instructional RAG to generate interactive and immersive visual instructions. By incorporating computer vision-based instructional RAG, the system can analyze images and generate corresponding instructions. With the integration of multimodal RAG for visual instruction, users can interact with the system using multiple modes of input, enhancing their understanding and retention of complex concepts.\n\nHow to use Vision Instruct\n\n1. Go to Vision Instruct tab.\n\n2. Click on upload option and upload the image.\n\n3. Give the prompt (Question).\n\n4. Click on Generate option."
    ]     
file = [
    "The file query system utilizes document-centric querying with RAG to retrieve relevant information from PDF documents. By incorporating PDF-based knowledge retrieval with RAG, the system can analyze the content of PDF files and generate answers to user queries. With the integration of file-based question answering with RAG, users can ask questions about the content of files and receive accurate and relevant responses.\n\nHow to use File Query\n\n1. Go to File Query tab.\n\n2. Click on upload option and upload the PDF File.\n\n3. Give the prompt (Question).\n\n4. Click on Generate option."
    ]     

tabs = st.tabs(tab_titles)
with tabs[0]:
    def lottie(anime="anime.json"):
        with open(anime, "r", encoding='UTF-8') as animation:
            return json.load(animation)
    animes = lottie()
  


    col1, col2 = st.columns(2, gap="large", vertical_alignment="center")
    with col2:
          st_lottie(animes, width=300, height=300)
    with col1:
        
     st.markdown("""
        <h4>Welcome to Spark AI!</h4>
        <p style="text-align: justify;">Unlock the power of AI-driven image and file analysis with our innovative application. Sparkis designed to simplify complex tasks, providing accurate and efficient results.</p>

"""
                , unsafe_allow_html=True)
    st.markdown("""<hr>
            
               """, unsafe_allow_html=True)
   
    st.image(image="slide1.webp")
    st.markdown("""<h4>Retrieval Augumented Generation</h4>
                <p style="text-align: justify;">The Retrieval-Augmented Generation (RAG) framework leverages hybrid retrieval-generation techniques to produce more accurate and informative responses. By combining the strengths of retrieval and generation models, RAG enables knowledge-augmented language generation, where relevant facts and information are seamlessly integrated into the generated text. This approach facilitates generative retrieval, 
                allowing the model to retrieve and generate text in a single, unified framework. Ultimately, RAG has the potential to revolutionize natural language processing and language generation, enabling the development of more sophisticated and knowledgeable AI systems.</p><hr>""", unsafe_allow_html=True)
                
    st.markdown("""        <h4>Advantages of the Spark AI</h4>
        <p style="text-align: justify;">It simplifies daily life tasks by using AI, generates the anlyzed data with in a minute. It saves the time by reading all data in files using AI-driven model.</p>""", unsafe_allow_html=True)
    st.image(image="advantage.png")
    st.markdown("""<hr>
        <h4>Explore Our Features - Get Started</h4>
        <h5>Vision Instruct</h5>
        <p style="text-align: justify;">It is used to query with images. It let us analyze the image data by using the llama model.</p>
                
                """, unsafe_allow_html=True)
    with st.expander("V I S I O N - I N S T R U C T"):
        st.write(vision[0])

    st.markdown("""
        <h5>File Query</h5>
        <p style="text-align: justify;">It is used to query with files. It let us analyze the files like PDF, TXT and so on by using the llama model.</p>
    """, unsafe_allow_html=True)
    with st.expander("F I L E - Q U E R Y"):
        st.write(file[0])

with tabs[1]:
    #upload file
    def img_analyze(img_analyze="img_analyze.json"):
        with open(img_analyze, "r", encoding='UTF-8') as f:
            return json.load(f)
    img_analyze = img_analyze()
    st_lottie(img_analyze)

    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
            # Show the uploaded image
            st.image(uploaded_file, caption='Uploaded Image')
            prompt = st.text_input('Enter the prompt')

            if st.button('Generate'):
                with st.spinner('Generating output...'):
                    if prompt:
                        output = generate(uploaded_file, prompt)
                    else:
                        output = generate(uploaded_file, 'What is in this picture?')
                st.subheader('Result:')
                st.write(output)

with tabs[2]:        
    def pdf_analyze(pdf_analyze="pdf_analyze.json"):
        with open(pdf_analyze, "r", encoding='UTF-8') as f:
            return json.load(f)
    pdf_analyze = pdf_analyze()
    st_lottie(pdf_analyze)   


    groq_api_key = os.getenv('GROQ_API_KEY')

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")


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


        if "vector_store" not in st.session_state:


            with tempfile.NamedTemporaryFile(delete=False) as temp_file:

                temp_file.write(pdf_file.read())

                pdf_file_path = temp_file.name

            st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
            
            st.session_state.loader = PyPDFLoader(pdf_file_path)

            st.session_state.text_document_from_pdf = st.session_state.loader.load()

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)

            st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)


    pdf_input_from_user = st.file_uploader("Upload the PDF file", type=['pdf'])


    if pdf_input_from_user is not None:
        with st.spinner('Generating output...'):
    
            if pdf_input_from_user is not None:

                create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
            
            else:
                
                st.write("Please upload a PDF file first")



    if "vector_store" in st.session_state:

        user_prompt = st.text_input("Enter Your Question related to the uploaded PDF")

        if st.button('Generate Analysis'):
                if user_prompt: 
                    if "vector_store" in st.session_state:

                        document_chain = create_stuff_documents_chain(llm, prompt)

                        retriever = st.session_state.vector_store.as_retriever()

                        retrieval_chain = create_retrieval_chain(retriever, document_chain)

                        response = retrieval_chain.invoke({'input': user_prompt})

                        st.write(response['answer'])

                    else:   

                        st.write("Please embed the document first by uploading a PDF file.")

                else:

                    st.error('Please write your prompt')

with tabs[3]:
    #upload file
    st.markdown("""
        <h4>About Spark AI</h4>
        <p style="text-indent: 60px; text-align: justify;"> Spark is an AI-powered application developed as part of the Applied Artificial Intelligence: Practical Implementations course  by TechSaksham Program, which is a CSR initiative by Microsoft and SAP, implemented by Edunet Foundation</p>
        <hr>""", unsafe_allow_html=True)
    col5, col6 = st.columns(2, gap="large", vertical_alignment="center")
    with col5:
        st.markdown("""        <ul> 
            <h3>Project Development Group Details</h3>
            <h4>Team Members</h4>
            <li>Sathvik Palivela</li>
            <li>Ravi Kiran Rayudu</li>          
        </ul>
        <ul>
            <h4>Mentor</h4>
            <li>Abdul Aziz Md</li>
        </ul>
        <br>""", unsafe_allow_html=True)
    with col6:
        def coding(coding = "coding.json"):
            with open(coding, 'r', encoding='UTF-8') as f:
                return json.load(f)
        icon = coding()
        st_lottie(icon, width=350, height=350)
    st.markdown("""<hr>
        <h4>Acknowledgements</h4>
        <p>We would like to extend our gratitude to: </p>
        <ul><li>TechSaksham Program, a CSR initiative by Microsoft and SAP</li>
            <li>Edunet Foundation for implementing the AI Practical Implementations course</li>
            <li>Aziz Sir for excellent guidance and mentorship</li></ul>
        <br>
        <h4>GitHub Repository</h4>
        <p>Check our github repository - <a href='https://github.com/SATHVIK-CONNECT/Project/tree/main'>Git Repo of Spark AI</a></p>
        <br> 
        <h4>Contact Us</h4>
        <p>For any queries or feedback, please reach out to us at <a>sathvikpalivela0@gmail.com</a>.
    """, unsafe_allow_html=True)
