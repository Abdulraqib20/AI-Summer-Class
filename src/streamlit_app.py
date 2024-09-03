import streamlit as st
import os
import sys
import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import time
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from utils.chat_memory import chat_memory_manager, generate_session_id
from prompts.instruction import INSTPROMPT
from config import appconfig

# Load environment variables
load_dotenv()

# Constants
MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "claude-3-5-sonnet",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]
GROQ_API_KEY = appconfig.groq_key
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.text_splitter = None

# Streamlit App Interface
st.title("LLM-Powered Document Query App")

# Sidebar for uploading documents
st.sidebar.title("Upload Your Documents")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt", "pdf", "docx"])

# Initialize session ID
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = generate_session_id()

# Select Model
model_choice = st.selectbox("Select a Model", MODELS)

# Initialize LLM client (Groq)
llm_client = Groq(model_choice, api_key=GROQ_API_KEY, temperature=0.1)

# Upload Documents
def upload_documents(uploaded_file):
    if uploaded_file is not None:
        with open(os.path.join("./data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Document uploaded successfully!")
        # Load the uploaded file into the document store
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index
    else:
        return None

index = upload_documents(uploaded_file)

# Display Query Input Field
query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if not index:
        st.error("Please upload a document before querying.")
    elif not query:
        st.error("Please enter a query.")
    else:
        st.info("Processing your query...")

        # Create custom prompt
        custom_prompt = PromptTemplate(INSTPROMPT)

        # Set up chat engine
        chat_engine = index.as_chat_engine(
            llm=llm_client, 
            similarity_top_k=5, 
            verbose=True, 
            system_prompt=custom_prompt
        )
        chat_engine = chat_memory_manager.apply_chat_memory(chat_engine, st.session_state['session_id'])

        # Process query
        start_time = time.time()
        response = chat_engine.chat(query)
        end_time = time.time()

        # Save chat history
        chat_memory_manager.add_message(st.session_state['session_id'], f"Human: {query}")
        chat_memory_manager.add_message(st.session_state['session_id'], f"AI: {response.response}")

        # Display response and timing
        st.write("### Response:")
        st.write(response.response)
        st.write(f"Query executed in {end_time - start_time:.2f} seconds.")

        # Display chat history
        st.write("### Chat History:")
        for role, message in chat_memory_manager.get_chat_history(st.session_state['session_id']):
            st.write(f"**{role.capitalize()}:** {message}")

# Clear Chat History Button
if st.button("Clear Chat History"):
    chat_memory_manager.clear_chat_history(st.session_state['session_id'])
    st.success("Chat history cleared!")
