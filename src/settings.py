from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


class ChromaSettings(BaseModel):
    path: str
    collection_name: str
    top_matches: int
    chunk_size: int
    chunk_overlap: int
    score_threshold: float


class OpenAiSettings(BaseModel):
    api_key: str
    no_answer_key: str
    model: str


class RagSettings(BaseModel):
    data_folder: Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter='__')

    chroma: ChromaSettings
    open_ai: OpenAiSettings
    rag: RagSettings


settings = Settings()
