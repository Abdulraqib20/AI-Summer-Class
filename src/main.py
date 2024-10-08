# import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from llama_index.llms.groq import Groq
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import TokenTextSplitter

# from src.config import appconfig
from src.config import appconfig
from dotenv import load_dotenv;load_dotenv()
from src.utils.chat_memory import chat_memory_manager, generate_session_id
from src.prompts.instruction import INSTPROMPT

print("Initializing LLM application...")



# Set GROQ_API_KEY = "your api key" in the .env file, then load it below
GROQ_API_KEY = appconfig.groq_key
print(GROQ_API_KEY)

models = [
    # "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "claude-3-5-sonnet",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]



"""
In llama-index, the LLM and embed_model can be set at any of 2 levels:
    - global seting with Settings (both llm and embed_model)
    - index level (embed_model only)
    - query engine level (llm only)
"""


Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Settings.llm = Groq(
#     models[0], 
#     api_key = GROQ_API_KEY,
#     temperature = 0.1
# )


def upload_doc(dir):

    print("Uploading documents...")
    documents = SimpleDirectoryReader(dir).load_data() 

    """You can apply splitting with global Settings"""
    Settings.text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
    index = VectorStoreIndex.from_documents(documents)

    """
    Or you can apply splitting at index level
    
        text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[text_splitter] # you can add any other transformation to this list
        )

    Other splitters you can play around with for different use cases, and lots more!
        "SentenceSplitter",
        "CodeSplitter",
        "HTMLNodeParser",
        "MarkdownNodeParser",
        "JSONNodeParser",
        "SentenceWindowNodeParser",
        "SemanticSplitterNodeParser",
        "NodeParser",
        "MetadataAwareTextSplitter",
        "UnstructuredElementNodeParser",
    """

    return index


def qa_engine(query: str, index, llm_client, choice_k=5, session_id=None):
    if not session_id:
        session_id = generate_session_id()
    
    custom_prompt = PromptTemplate(INSTPROMPT)
    
    query_engine = index.as_query_engine(
        llm=llm_client, 
        similarity_top_k=choice_k, 
        verbose=True,
        system_prompt=custom_prompt
    )
    
    query_engine_with_memory = chat_memory_manager.apply_chat_memory(query_engine, session_id)
    
    chat_history = chat_memory_manager.get_chat_history(session_id)
    response = query_engine_with_memory.query(query, chat_history=chat_history)
    
    chat_memory_manager.add_message(session_id, "human", query)
    chat_memory_manager.add_message(session_id, "ai", response.response)
    
    return response, session_id



if __name__ == "__main__":
    index = upload_doc("./data")
    session_id = None
    
    while True:
        query = input("Ask me anything (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        model = input("Enter model code: ")
        llm_client = Groq(model, api_key=GROQ_API_KEY, temperature=0.1)
        
        response, session_id = qa_engine(query, index, llm_client, session_id=session_id)
        
        print(f"Response: {response}")
        print(f"Session ID: {session_id}")

