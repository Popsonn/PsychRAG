import os
import re
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from huggingface_hub import snapshot_download
import tempfile
from dotenv import load_dotenv
import shutil

load_dotenv()
groq_api_key = st.secrets.secrets.GROQ_API_KEY
hugging_face_api_key = st.secrets.secrets.hugging_face_api_key
langchain_api_key = st.secrets.secrets.langchain_api_key

def remove_think_tags(text):
    # Use regex to remove <think> tags and their contents
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def extract_cot(text):
    # Use regex to extract the content within <think> tags
    cot_text = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    return "\n".join(cot_text).strip()

st.title("PsychRAG with Deepseek-R1-Distill-Llama-70b")

llm = ChatGroq(model="Deepseek-R1-Distill-Llama-70b", api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
"""
You are an AI expert trained to assist in analyzing biomedical literature,
specifically focusing on topics related to suicide, suicide ideation, mental health,
self-harm, depression, and psychological distress. Your role is to provide clear, 
evidence-based answers derived from PubMed articles and research findings.

Answer the questions based on the provided context only.
Provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {question}

Answer:
""")

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2')

import os
import tempfile
from huggingface_hub import snapshot_download
import shutil
import streamlit as st
from langchain_community.vectorstores import FAISS

@st.cache_resource
def load_vector_store():
    vector_store_path = os.path.join(os.getcwd(), "psychrag-vectorstore")
    print(f"Target vector store path: {vector_store_path}")
    
    try:
        # If vector store doesn't exist locally, download it
        if not os.path.exists(vector_store_path):
            print("Downloading from Hugging Face...")
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the entire repository
                repo_path = snapshot_download(
                    repo_id="Popson/psychrag-vectorstore",
                    repo_type="dataset",
                    local_dir=temp_dir,
                    token=None  # Add your token here if it's a private repo
                )
                
                # Create the target directory
                os.makedirs(vector_store_path, exist_ok=True)
                
                # List contents of downloaded repo
                print(f"Downloaded files: {os.listdir(repo_path)}")
                
                # Copy files instead of moving them
                for file_name in os.listdir(repo_path):
                    src = os.path.join(repo_path, file_name)
                    dst = os.path.join(vector_store_path, file_name)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        print(f"Copied {file_name} to {dst}")
        
        # Verify the required files exist
        required_files = ['index.faiss', 'index.pkl']
        existing_files = os.listdir(vector_store_path)
        print(f"Files in vector store directory: {existing_files}")
        
        missing_files = [f for f in required_files if f not in existing_files]
        if missing_files:
            raise FileNotFoundError(f"Missing required files: {missing_files}")
        
        # Load the vector store
        print("Loading FAISS index...")
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Successfully loaded FAISS index")
        return vector_store
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents at target path: {os.listdir(vector_store_path) if os.path.exists(vector_store_path) else 'Directory does not exist'}")
        raise Exception(f"Failed to load vector store: {str(e)}")

# Initialize the RAG chain
def init_rag_chain():
    vector_store = load_vector_store()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt  
        }
    )
    return qa_chain

# Initialize the chain
qa_chain = init_rag_chain()

# Create the query input
question = st.text_input("Enter your question:")

if question:
    with st.spinner("Searching through the articles..."):
        result = qa_chain.invoke({"query": question})
        
        # Extract and clean the CoT
        cot_text = extract_cot(result['result'])
        cleaned_result = remove_think_tags(result['result'])
        
        # Display the CoT in an expander
        with st.expander("Chain of Thought (CoT)"):
            st.write(cot_text)
        
        # Display the answer
        st.write("### Answer")
        st.write(cleaned_result)
        
        # Display sources if you want them
        if 'source_documents' in result:
            st.write("### Sources")
            for i, doc in enumerate(result['source_documents']):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
