import os, tempfile, traceback
from typing import List, Literal, Any
from fastapi import FastAPI, Request, Form, UploadFile, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    StorageContext, 
    load_index_from_storage
)
from cachetools import TTLCache


from src.utils.helpers import (
    upload_files, QueryEngineError, init_chroma, get_vector_store, get_kb_size
)
from src.exceptions.operationshandler import system_logger
from src.utils.models import LLMClient
from dotenv import load_dotenv;load_dotenv()
from src.main import qa_engine
from src.config.appconfig import groq_key
from src.utils.chat_memory import chat_memory_manager, generate_session_id

app = FastAPI()

# class EmbeddingState:
#     """
#     Implementation of dependency injection intended for working locally with \
#         embeddings via in-session storage. It allows you to have session-wide access \
#             to embeddings across the different endpoints. \
#                 This is not ideal for production.
#     """

#     def __init__(self):
#         self.embedding = None

#     def get_embdding_state():
#         return state

# state = EmbeddingState()
        

@app.get('/healthz')
async def health():
    return {
        "application": "Simple LLM API",
        "message": "running succesfully"
    }
    

@app.post('/upload')
async def process(
    projectUuid: str = Form(...),
    files: List[UploadFile] = None
):

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            _uploaded = await upload_files(files, temp_dir)

            if _uploaded["status_code"]==200:
                documents = SimpleDirectoryReader(temp_dir).load_data()
                
                
                # embedding = VectorStoreIndex.from_documents(documents)
                # embedding_save_dir = f"src/week_3/day_4_robust_rag/vector_db/{projectUuid}"
                # os.makedirs(embedding_save_dir, exist_ok=True)
                # embedding.storage_context.persist(persist_dir=embedding_save_dir)
                
                
                collection_name = projectUuid
                chroma_collection = init_chroma(collection_name=collection_name, path=r"src\chromadb")
                
                print(f"Existing collection size ::: {get_kb_size(chroma_collection)}")
                
                vector_store = get_vector_store(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                embedding = VectorStoreIndex.from_documents(
                    documents=documents,
                    storage_context=storage_context
                )
                
                print(f"Collection size after new Embeddings ::: {get_kb_size(chroma_collection)}")
                
                return {
                    'detail': 'Embeddings generated succesfully',
                    'status_code': 200
                }
            else:
                return _uploaded

        
    except Exception as e:
        print(traceback.format_exc())
        return {
            'detail': f'Could not generate embeddings: {e}',
            'status_code': 500
        }


@app.post('/generate')
async def generate_chat(
    request: Request
):

    # Parse the incoming request JSON
    query = await request.json()
    model = query["model"]
    temperature = query["temperature"]
    session_id = query.get("session_id")
    
    if not session_id:
        session_id = generate_session_id()

    # Debugging: Print the Groq API key to ensure it exists
    print("Groq API Key:", groq_key)
    
    init_client = LLMClient(
        groq_api_key = groq_key,
        secrets_path="./service_account.json",
        temperature=temperature,
        max_output_tokens=512
    )
    
    llm_client = init_client.map_client_to_model(model)
    
    # embedding_path = f'src/week_3/day_4_robust_rag/vector_db/{query["projectUuid"]}'
    # storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
    # embedding = load_index_from_storage(storage_context)
    
    chroma_collection = init_chroma(collection_name=query["projectUuid"], path=r"src\chromadb")
    collection_size = get_kb_size(chroma_collection)
    print(f"Retrieved collection size ::: {collection_size}")
    
    vector_store = get_vector_store(chroma_collection=chroma_collection)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embedding = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        # storage_context=storage_context
    )
    
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

    print(f"Retrieving top {choice_k} chunks from knowledge base ::: {collection_size}")

    try:
        chat_engine = embedding.as_chat_engine(
            llm=llm_client,
            similarity_top_k=choice_k,
            verbose=True,
        )
        chat_engine_with_memory = chat_memory_manager.apply_chat_memory(chat_engine=chat_engine, session_id=session_id)
        
        response = chat_engine_with_memory.chat(query["question"])
        
        chat_memory_manager.add_message(session_id, "human", query["question"])
        chat_memory_manager.add_message(session_id, "ai", response.response)
        
        return PlainTextResponse(content=response.response, status_code=200, headers={"X-Session-ID": session_id})
        
        # # Generate the response using the QA engine
        # response = qa_engine(
        #     query["question"], 
        #     embedding,
        #     llm_client,
        #     choice_k=choice_k
        # )

        # print(response.response)
        # return PlainTextResponse(content=response.response, status_code=200)
    
    except Exception as e:
        message = f"An error occurred where {model} was trying to generate a response: {e}"
        system_logger.error(message, exc_info=1)
        raise QueryEngineError(message)
    

@app.post('/clear_chat_history')
async def clear_chat_history(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    chat_memory_manager.clear_chat_history(session_id)
    return {"message": "Chat history cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)

