import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import uuid
import tempfile
from typing import List
import json
import traceback
from typing import List
import tempfile

# Import necessary modules from your existing codebase
from src.utils.helpers import upload_files, init_chroma, get_vector_store, get_kb_size, allowed_file
from src.utils.models import LLMClient
from src.utils.chat_memory import chat_memory_manager, generate_session_id
from src.config.appconfig import groq_key
from src.prompts.instruction import INSTPROMPT
from src.style import load_css

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext, 
    load_index_from_storage
)
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from cachetools import TTLCache

# Constants
UPLOAD_DIR = "uploaded_files"
CHAT_HISTORY_FILE = "chat_history.json"

# Ensure directories for file uploads and chat history
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@st.cache_data
# Function to process uploaded files and generate embeddings
def process_files(files: List[st.runtime.uploaded_file_manager.UploadedFile], project_uuid: str):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_files = []
            
            for file in files:
                if file is not None and allowed_file(file.name):
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                    processed_files.append(file.name)
                else:
                    st.warning(f"Skipped file {file.name}: Not an allowed file type.")
            
            if not processed_files:
                return "No valid files were uploaded."
            
            documents = SimpleDirectoryReader(temp_dir).load_data()
            
            # Initialize Chroma collection
            chroma_collection = init_chroma(collection_name=project_uuid, path=r"src\chromadb")
            
            st.write(f"Existing collection size: {get_kb_size(chroma_collection)}")
            
            vector_store = get_vector_store(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            embedding = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context
            )
            
            st.write(f"Collection size after new Embeddings: {get_kb_size(chroma_collection)}")
            
            return f"Embeddings generated successfully for {', '.join(processed_files)}"
        
    except Exception as e:
        st.error(f"Could not generate embeddings: {str(e)}")
        return f"Error: {str(e)}"
    
@st.cache_resource
# Function to generate chat responses
def generate_chat_response(question: str, model: str, temperature: float, project_uuid: str, session_id: str):
    try:
        init_client = LLMClient(
            groq_api_key=groq_key,
            secrets_path="./service_account.json",
            temperature=temperature,
            max_output_tokens=512
        )
        
        llm_client = init_client.map_client_to_model(model)
        
        chroma_collection = init_chroma(collection_name=project_uuid, path=r"src\chromadb")
        collection_size = get_kb_size(chroma_collection)
        
        vector_store = get_vector_store(chroma_collection=chroma_collection)
        embedding = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        def determine_choice_k(collection_size):
            if collection_size > 150:
                return 40
            elif collection_size > 50:
                return 20
            elif collection_size > 20:
                return 10
            else:
                return 5

        choice_k = determine_choice_k(collection_size)
        
        custom_prompt = PromptTemplate(INSTPROMPT)
        
        chat_engine = embedding.as_chat_engine(
            llm=llm_client, 
            similarity_top_k=choice_k, 
            verbose=True,
            system_prompt=custom_prompt
        )
        
        chat_engine_with_memory = chat_memory_manager.apply_chat_memory(chat_engine, session_id)
        
        chat_history = chat_memory_manager.get_chat_history(session_id)
        response = chat_engine_with_memory.chat(
            message=question,
            chat_history=chat_history
        )
        
        chat_memory_manager.add_message(session_id, "human", question)
        chat_memory_manager.add_message(session_id, "ai", response.response)
        
        return response.response
    
    except Exception as e:
        st.error(f"An error occurred while generating the response: {str(e)}")
        return f"Error: {str(e)}"
    

# Function to clear chat history
def clear_chat_history(session_id: str):
    chat_memory_manager.clear_chat_history(session_id)
    return "Chat history cleared successfully"


# Function to load chat history from file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


# Function to save chat history to file
def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(messages, f)
        
        

# Main app logic
def main():
    st.set_page_config(page_title="AI Chat Assistant", layout="wide", page_icon="📊")
    load_css()
    st.title("AI Chat Assistant")

    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if 'project_uuid' not in st.session_state:
        st.session_state.project_uuid = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Settings")
        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    result = process_files(uploaded_files, st.session_state.project_uuid)
                    st.success(result)

        model = st.selectbox("Select Model", [
            "llama-3.1-70b-versatile", 
            "llama-3.1-8b-instant", 
            "mixtral-8x7b-32768",
            "claude-3-5-sonnet",
            "claude-3-haiku",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

        if st.button("Clear Chat History"):
            clear_chat_history(st.session_state.session_id)
            st.session_state.messages = []
            st.success("Chat history cleared")

    # Main chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input for the chat
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chat_response(
                    prompt, 
                    model, 
                    temperature, 
                    st.session_state.project_uuid,
                    st.session_state.session_id
                )
            st.markdown(response)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save chat history after each interaction
        save_chat_history(st.session_state.messages)

if __name__ == "__main__":
    main()