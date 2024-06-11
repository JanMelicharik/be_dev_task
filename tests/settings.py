from pydantic import BaseModel


class ChromaSettings(BaseModel):
    path: str = "test_path"


class OpenAiSettings(BaseModel):
    api_key: str = "test_api"


class Settings(BaseModel):
    chroma: ChromaSettings = ChromaSettings()
    open_ai: OpenAiSettings = OpenAiSettings()


settings = Settings()
