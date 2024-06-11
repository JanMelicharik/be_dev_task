from openai import OpenAI
from src.settings import settings
from langchain_core.documents.base import Document
from src.open_ai.template import prompt_template


class OpenAiClient:
    def __init__(self, client: OpenAI, model: str, no_answer_key: str):
        self.client = client
        self.model = model
        self.no_answer_key = no_answer_key

    def generate_response(self, query: str, context_documents: list[Document], verbose: bool = True) -> str:
        content = prompt_template(context_documents, query, settings.open_ai.no_answer_key, verbose)
        stream = self.client.chat.completions.create(
            model=settings.open_ai.model,
            messages=[{"role": "user", "content": content}],
            stream=True,
        )
        answer = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                answer += chunk.choices[0].delta.content

        return answer
