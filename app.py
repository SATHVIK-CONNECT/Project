import streamlit as st
from groq import Groq
import base64
import os
 
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
        <button style="padding: 7px 25px; background: linear-gradient(to right, #1da5f2, purple); border: none; border-radius: 7px;">Vision Instruct</button>
        <br/><br>
        <h5>File Query</h5>
        <p style="text-align: justify;">It is used to query with files. It let us analyze the files like PDF, TXT and so on by using the llama model.</p>
        <button style="padding: 7px 25px; background: linear-gradient(30deg, #1da5f2, purple); border: none; border-radius: 7px;">File Query</button>
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
    uploaded_file = st.file_uploader('Upload an image', type=['pdf', 'doc', 'docx'])

    if uploaded_file is not None:
            # Show the uploaded image
            # st.image(uploaded_file, caption='Uploaded Image')
            prompt = st.text_input('Enter Query')

            if st.button('Click'):
                with st.spinner('Generating output...'):
                    if prompt:
                        output = generate_file(uploaded_file, prompt)
                    else:
                        output = generate(uploaded_file, 'What is in this picture?')
                st.subheader('Output:')
                st.write(output)

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

