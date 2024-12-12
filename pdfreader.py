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
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
api_key=#Enter API KEY
os.environ["GOOGLE_API_KEY"] = #Enter API KEY

summary_prompt="""
Summarize the following {element_type}:
{element}
"""
summary_chain= LLMChain(
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key,max_tokens=1024),
    prompt = PromptTemplate.from_template(summary_prompt)
    
)

output_path="/Users/priyanshiranawat/Multimodal-healthcare/Images"
def read_pdf(pdf_path):
    raw_pdf_elements= partition_pdf(
    filename=pdf_path,
    extract_images_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy='by_title',
    max_characters=4000,
    new_after_n_chars=2000,
    extract_image_block_output_dir=output_path
)
    return raw_pdf_elements

def summarize_pdf_elements(raw_pdf_elements, llm_chain):
    
    text_elements = []
    table_elements = []
    text_summaries = []
    table_summaries = []

    for element in raw_pdf_elements:
        if 'CompositeElement' in repr(element):
            text_elements.append(element.text)
            summary = llm_chain.run({'element_type': 'text', 'element': element.text})
            text_summaries.append(summary)
        elif 'Table' in repr(element):
            table_elements.append(element.text)
            summary = llm_chain.run({'element_type': 'table', 'element': element.text})
            table_summaries.append(summary)
    
    return text_elements, table_elements, text_summaries, table_summaries


def summarize_images(output_path, api_key):
    """
    Encodes and summarizes images from a specified directory.

    Args:
        output_path (str): Path to the directory containing images.
        api_key (str): API key for the ChatGoogleGenerativeAI model.

    Returns:
        tuple: A tuple containing lists of encoded images and their summaries.
    """
    image_elements = []
    image_summaries = []

    def encode_image(image_path):
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None


    def summarize_image(encoded_image, api_key):
        prompt = [
            SystemMessage(content="You are a bot that is good at analyzing images"),
            HumanMessage(content=[
                {
                    "type": 'text',
                    "text": "Describe the contents of this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    },
                },
            ])
        ]
        response = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key, max_tokens=1024).invoke(prompt)
        time.sleep(2.2)
        return response.content

    for file_name in os.listdir(output_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(output_path, file_name)
            encoded_image = encode_image(image_path)
            if encode_image:
                image_elements.append(encoded_image)
                summary = summarize_image(encoded_image, api_key)
                image_summaries.append(summary)

    return image_elements, image_summaries

from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings()



def create_vectorstore(text_elements, text_summaries, table_elements, table_summaries, image_elements, image_summaries, embedding_model):

    documents = []
    retrieve_contents = []

    # Helper function to create a document
    def create_document(content, content_type, original_content):
        doc_id = str(uuid.uuid4())
        doc = Document(
            page_content=content,
            metadata={
                'id': doc_id,
                'type': content_type,
                'original_content': original_content
            }
        )
        retrieve_contents.append((doc_id, original_content))
        documents.append(doc)

    # Add text documents
    for e, s in zip(text_elements, text_summaries):
        create_document(s if isinstance(s, str) else str(s), 'text', e)

    # Add table documents
    for e, s in zip(table_elements, table_summaries):
        create_document(s if isinstance(s, str) else str(s), 'table', e)

    # Add image documents
    for e, s in zip(image_elements, image_summaries):
        create_document(s if isinstance(s, str) else str(s), 'image', e)

    # Create VectorStore
    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding_model)

    return vectorstore, retrieve_contents

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


def answer(question,vectorstore,qa_chain):
    relevant_docs= vectorstore.similarity_search(question)
    context=""
    relevant_images=[]
    for d in relevant_docs:
        if d.metadata['type']=='text':
            context += '[text]' + d.metadata['original_content']
        elif d.metadata['type']=='table':
            context += '[table]' +d.metadata['original_content']
        elif d.metadata['type'] == 'image':
            context += '[image]' + d.page_content
            if os.path.exists(d.metadata['original_content']):
                relevant_images.append(d.metadata['original_content'])
            else:
                print(f"Warning: Image file not found - {d.metadata['original_content']}")
            relevant_images.append(d.metadata['original_content'])
    result = qa_chain.run({'context': context, 'question': question})
    return result, relevant_images   









