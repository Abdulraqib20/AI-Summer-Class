import os, tempfile, traceback
from typing import List, Literal, Any
from fastapi import FastAPI, Request, Form, UploadFile, Depends
from fastapi.responses import PlainTextResponse
from src.week_3.day_4_robust_rag.main import *
from src.week_3.day_4_robust_rag.utils.helpers import system_logger, upload_files, QueryEngineError
from src.week_3.day_4_robust_rag.utils.models import LLMClient
from dotenv import load_dotenv;load_dotenv()
from src.config.appconfig import groq_key

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
    files: List[UploadFile] = None,
    # state: EmbeddingState = Depends(EmbeddingState.get_embdding_state)
):

    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            _uploaded = await upload_files(files, temp_dir)

            if _uploaded["status_code"]==200:

                documents = SimpleDirectoryReader(temp_dir).load_data()
                embedding = VectorStoreIndex.from_documents(documents)

                embedding_save_dir = f"src/week_3/day_4_robust_rag/vector_db/{projectUuid}"
                os.makedirs(embedding_save_dir, exist_ok=True)
                
                embedding.storage_context.persist(persist_dir=embedding_save_dir)
                
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
    request: Request,
    # state: EmbeddingState = Depends(EmbeddingState.get_embdding_state)
):

    # Parse the incoming request JSON
    query = await request.json()
    model = query["model"]
    temperature = query["temperature"]

    # Debugging: Print the Groq API key to ensure it exists
    print("Groq API Key:", groq_key)
    
    init_client = LLMClient(
        groq_api_key = groq_key,
        secrets_path="./service_account.json",
        # secrets_path=r"src\config\service_account.json",
        temperature=temperature,
        max_output_tokens=512
    )
    
    llm_client = init_client.map_client_to_model(model)
    
    embedding_path = f'src/week_3/day_4_robust_rag/vector_db/{query["projectUuid"]}'
    storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
    embedding = load_index_from_storage(storage_context)

    try:
        # Generate the response using the QA engine
        response = qa_engine(
            query["question"], 
            embedding,
            llm_client,
        )

        print(response.response)
        return PlainTextResponse(content=response.response, status_code=200)
    
    except Exception as e:
        message = f"An error occured where {model} was trying to generate a response: {e}",
        system_logger.error(
            message,
            exc_info=1
        )
        raise QueryEngineError(message)


if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)

