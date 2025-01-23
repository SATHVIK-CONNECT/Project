import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from groq import Groq
import base64
import os
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile
 
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

# Function to generate caption
def generate_file(uploaded_file, prompt):
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
                            'url': uploaded_file,
                        },
                    },
                ],
            }
        ],
        model='llama-3.2-90b-vision-preview',
    )
    return chat_completion.choices[0].message.content

# Streamlit App
st.title("SparkAI")

tab_titles = [
    "Home",
    "Vision Instruct",
    "File Query",
    "About",
]

tabs = st.tabs(tab_titles)

with tabs[0]:
    st.markdown("""
        <h4>Welcome to Intellect!</h4>
        <p style="text-align: justify;">Unlock the power of AI-driven image and file analysis with our innovative application. Sparkis designed to simplify complex tasks, providing accurate and efficient results.</p>
        <h4>Advantages of the Intellect</h4>
        <p style="text-align: justify;">It simplifies daily life tasks by using AI, generates the anlyzed data with in a minute. It saves the time by reading all data in files using AI-driven model.</p>
        <h4>Explore Our Features - Get Started</h4>
        <h5>Vision Instruct</h5>
        <p style="text-align: justify;">It is used to query with images. It let us analyze the image data by using the llama model.</p>
        <button style="padding: 7px 25px; background: linear-gradient(35deg, #1da5f2, purple); border: none; border-radius: 7px;">Vision Instruct</button>
        <br/><br>
        <h5>File Query</h5>
        <p style="text-align: justify;">It is used to query with files. It let us analyze the files like PDF, TXT and so on by using the llama model.</p>
        <button style="padding: 7px 25px; background: linear-gradient(35deg, #1da5f2, purple); border: none; border-radius: 7px;">File Query</button>
    """, unsafe_allow_html=True)

with tabs[1]:
    #upload file
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
            # Show the uploaded image
            st.image(uploaded_file, caption='Uploaded Image')
            prompt = st.text_input('Enter Query')

            if st.button('Click'):
                with st.spinner('Generating output...'):
                    if prompt:
                        output = generate(uploaded_file, prompt)
                    else:
                        output = generate(uploaded_file, 'What is in this picture?')
                st.subheader('Output:')
                st.write(output)

with tabs[2]:
    #upload file
    load_dotenv()


    groq_api_key = os.getenv('GROQ_API_KEY')


    st.markdown("<h2 style='text-align: center;'>PDF Insights: Interactive Q&A with Llama3 & Groq API</h2>", unsafe_allow_html=True)


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

        if st.button("Create the Vector DB from the uploaded PDF file"):
            
            if pdf_input_from_user is not None:
                
                create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
                
                st.success("Vector Store DB for this PDF file Is Ready")
            
            else:
                
                st.write("Please upload a PDF file first")



    if "vector_store" in st.session_state:

        user_prompt = st.text_input("Enter Your Question related to the uploaded PDF")

        if st.button('Submit Prompt'):

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
        <h4>About Intellect</h4>
        <p style="text-indent: 60px; text-align: justify;"> Spark is an AI-powereed application developed as part of the Applied Artificial Intelligence: Practical Implementations course  by TechSaksham Program, which is a CSR initiative by Micrososft and SAP, implemented by Edunet Foundation</p>
        <br>
        <ul> 
            <p>Here are the details of the Project development group</p>
            <h4>Team Members</h4>
            <li>Sathvik</li>
            <li>Ravi Kiran</li>
            <li>Keeravani</li>
            <li>Prem Kumar</li>
        </ul>
        <ul>
            <h4>Mentor</h4>
            <li>Abdul Aziz Md</li>
        </ul>
        <br>
        <h4>Acknowledgements</h4>
        <p>We would like to extend our gratitude to: </p>
        <ul><li>TechSaksham Program, a CSR initiative by Microsoft and SAP</li>
            <li>Edunet Foundation for implementing the AI Practical Implementations course</li>
            <li>Aziz Sir for excellent guidance and mentorship</li></ul>
        <br>
        <h4>Contact Us</h4>
        <p>For any queries or feedback, please reach out to us at <a>sathvikpalivela0@gmail.com</a>. 

    """, unsafe_allow_html=True)

