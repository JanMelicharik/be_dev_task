# Retrieval-Augmented Generation (RAG) API

A simple API that accepts an input question and returns an answer using OpenAI model with the help of ChromaDB.

## Getting started

You need to have Python 3.12 installed. You can use [pyenv](https://github.com/pyenv/pyenv) for python version
management.

You also need to have [Poetry](https://python-poetry.org/docs/#installing-with-pipx) installed. 

After that you can install the dependencies:

```bash
poetry install
```

And get into the virtual environment:

```bash
poetry shell
```

You will also need to set the following environment variables. You can create a `.env` file in the root of the project with the following content:

```env
CHROMA__PATH=chroma
CHROMA__COLLECTION_NAME=rag_app
CHROMA__TOP_MATCHES=3
CHROMA__CHUNK_SIZE=500
CHROMA__CHUNK_OVERLAP=200
CHROMA__SCORE_THRESHOLD=0.5

OPEN_AI__API_KEY=
OPEN_AI__NO_ANSWER_KEY=OOPSIE_WOOPSIE
OPEN_AI__MODEL=gpt-3.5-turbo

RAG__DATA_FOLDER=data/
```

You can then run the following command to load the environment variables:

```bash
export $(xargs < .env) 
```

You will need to provide your OpenAI API key for the app to work. Feel free to play around with the configuration.

## Running the API

You can run the API with the following command:

```bash
fastapi run src/main.py
```

## API Endpoints

- `/` - health check endpoint
- `/query` - query endpoint that accepts a question and returns an answer

## Examples

Asking a question that the model can answer:

```bash
curl -G "http://localhost:8000/query" --data-urlencode "question=Why did we suspend sales in Italy?"

>> {"answer":"Answer: \n\nWe suspended sales in Italy due to the unstable economic environment... Source: data/no_sale_countries.md"}
```

Asking a question that the model can't answer:

```bash
curl -G "http://localhost:8000/query" --data-urlencode "question=How are you?"

>> {"answer":"Unfortunately, I don't have an answer for that."}
```