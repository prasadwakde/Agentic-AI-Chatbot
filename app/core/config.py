from pydantic_settings import BaseSettings

# Easier to add more configs here
class Settings(BaseSettings):
    groq_api_key: str | None = None
    tavily_api_key: str | None = None
    openai_api_key: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
