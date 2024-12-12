import streamlit as st
import pdfreader
import os
#from langchain.vectorstores import FAISS
import os
import uuid
import base64
from IPython import display
from unstructured.partition.pdf import partition_pdf
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
#from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from PIL import Image  # For image handling
from io import BytesIO  # For handling byte streams

from langchain_huggingface import HuggingFaceEmbeddings

api_key= #ENTER API KEY
os.environ["GOOGLE_API_KEY"] = #----------ENTER YOUR API KEY-------------------
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings()

summary_prompt="""
Summarize the following {element_type}:
{element}
"""
summary_chain= LLMChain(
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key,max_tokens=1024),
    prompt = PromptTemplate.from_template(summary_prompt)
    
)
prompt_template="""
You are a Vet doctor and expert in analysing dog's health.
Answer the following based on following context which can include text,
images and tables.
{context}
Question:{question}
Don't answer if you are not sure and decline to answer the question and respond, Sorry I don't have sufficient information to answer this.
Just return the helpful answer in detail. 
Answer:
"""

qa_chain= LLMChain(
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key,max_tokens=1024),
    prompt = PromptTemplate.from_template(prompt_template)
    
)

def main():
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    st.title("PDF Processing and Query System")
    st.sidebar.title("Upload PDF and Process")

    # Step 1: Upload PDF
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    output_path = "/Users/priyanshiranawat/Multimodal-healthcare/Images"

    if uploaded_file:
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Step 2: Process PDF on Button Click
        if st.sidebar.button("Process PDF"):
            # Call the PDF processing functions
            raw_pdf_elements = pdfreader.read_pdf(pdf_path)
            text_elements, table_elements, text_summaries, table_summaries = pdfreader.summarize_pdf_elements(raw_pdf_elements, summary_chain)
            image_elements, image_summaries = pdfreader.summarize_images(output_path, api_key)
            global vectorstore  # Make vectorstore accessible globally
            vectorstore, retrieve_contents = pdfreader.create_vectorstore(
                text_elements, text_summaries, table_elements, table_summaries,
                image_elements, image_summaries, hf
            )
            st.session_state['vectorstore']=vectorstore
            st.sidebar.success("Processing is done.")

            # Step 3: Query Input and Output
        if st.session_state['vectorstore']:
            print("Inside vectorstore global")
            if 'query' not in st.session_state:
                st.session_state.query=''
                print('No query')
            query = st.text_area("Ask your question:",value=st.session_state.query ,key="query")
            print(query)
            btn = st.button("Generate Output")
            if btn:
                print("Generating output")
                if query:
                    result, relevant_images = pdfreader.answer(query, st.session_state['vectorstore'], qa_chain)
                    st.subheader("Output:")
                    print(result)
                    st.write(result)

                    # Display relevant images
                    if relevant_images:
                        st.subheader("Relevant Images:")
                        for encoded_image in relevant_images:
                            try:
                                # Decode base64 string to image bytes
                                image_bytes = base64.b64decode(encoded_image)
                                image = Image.open(BytesIO(image_bytes))
                                st.image(image, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")

main()
