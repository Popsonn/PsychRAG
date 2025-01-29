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

load_dotenv()

import re

def remove_think_tags(text):
    # Use regex to remove <think> tags and their contents
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def extract_cot(text):
    # Use regex to extract the content within <think> tags
    cot_text = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    return "\n".join(cot_text).strip()

groq_api_key = os.getenv("GROQ_API_KEY")
hugging_face_api_key = os.getenv("hugging_face_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
langchain_api_key = os.getenv("langchain_api_key")

st.title("PsychRAG with Deepseek-R1-Distill-Llama-70b")

llm = ChatGroq(model="Deepseek-R1-Distill-Llama-70b")

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

@st.cache_resource  # This will cache the vector store load
def load_vector_store():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = snapshot_download(
             repo_id="Popson/psychrag-vectorstore",
            repo_type="dataset",
            local_dir=temp_dir
        )

    #vector_store_path = os.path.join(os.getcwd(), "pubmed_suicide_vectorstore")
    #os.makedirs(vector_store_path, exist_ok=True)
        return FAISS.load_local(repo_path, embeddings, allow_dangerous_deserialization=True)

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