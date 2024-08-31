from pydantic_settings import BaseSettings
from src.config import appconfig
from typing import Optional

class Settings(BaseSettings):
    """
    This class generates the BaseSettings for the FastAPI LLM application.
    It contains project definitions, environment configuration,
    and other application settings.
    """
    
    # API Configuration
    API_STR: str = '/api/v1' if appconfig.Env != 'development' else '/dev/api/v2'
    VERSION: str = '3.0.2'
    PROJECT_NAME: str = 'AI Server'
    ENV: str = appconfig.Env

    # API Keys
    GROQ_API_KEY: Optional[str] = appconfig.groq_key
    PINECONE_API_KEY: Optional[str] = appconfig.pinecone_key
    LANGCHAIN_API_KEY: Optional[str] = appconfig.langchain_key
    QDRANT_API_KEY: Optional[str] = appconfig.qdrant_key
    OPENAI_API_KEY: Optional[str] = appconfig.openai_api_key

    # Application Port
    APP_PORT: int = 5000

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Returns the Settings object."""
    return Settings()
