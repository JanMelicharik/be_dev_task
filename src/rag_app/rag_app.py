from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from src.open_ai.open_ai import OpenAiClient
from src.chroma_db.chroma_db import ChromaClient
from src.settings import settings


class RagApp:
    def __init__(self):
        self.chroma_client = ChromaClient(
            client=Chroma(
                collection_name=settings.chroma.collection_name,
                embedding_function=OpenAIEmbeddings(openai_api_key=settings.open_ai.api_key),
                persist_directory=settings.chroma.path,
            ),
            chunk_size=settings.chroma.chunk_size,
            chunk_overlap=settings.chroma.chunk_overlap,
            search_top_matches=settings.chroma.top_matches,
            score_threshold=settings.chroma.score_threshold,
        )
        self.openai_client = OpenAiClient(
            client=OpenAI(api_key=settings.open_ai.api_key),
            model=settings.open_ai.model,
            no_answer_key=settings.open_ai.no_answer_key,
        )

    def setup_chroma(self):
        self.chroma_client.load_local_resources(settings.rag.data_folder)

    def rag_query(self, query: str):
        matches = self.chroma_client.search_similar(query)
        answer = self.openai_client.generate_response(query, matches)
        if answer == self.openai_client.no_answer_key:
            return {"answer": "Unfortunately, I don't have an answer for that."}

        return {"answer": answer}
