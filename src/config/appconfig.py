from dotenv import load_dotenv
import os

load_dotenv()

Env = os.getenv("PYTHON_ENV", "development")
groq_key = os.getenv("GROQ_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")
qdrant_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
