from fastapi import FastAPI

from src.rag_app.rag_app import RagApp


rag_app = RagApp()
rag_app.setup_chroma()

app = FastAPI()


@app.get("/")
async def health_check():
    return {"message": "OK"}


@app.get(path="/query")
async def query(question: str):
    if not question:
        return {"answer": "Please provide a question."}
    return rag_app.rag_query(question)
