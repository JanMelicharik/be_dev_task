from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from structlog import BoundLogger, get_logger


class ChromaClient:
    ALLOWED_EXTENSIONS = ["md"]

    def __init__(
        self,
        client: Chroma,
        chunk_size: int,
        chunk_overlap: int,
        search_top_matches: int,
        score_threshold: float,
        logger: BoundLogger | None = None,
    ) -> None:
        self.client = client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        self.search_top_matches = search_top_matches
        self.score_threshold = score_threshold

        if self.logger is None:
            self.logger = get_logger(component="ChromaClient")

    def search_similar(self, query: str) -> list[Document]:
        self.logger.info("search_similar", status="started", query=query, top_matches=self.search_top_matches)
        matches = self.client.similarity_search_with_score(query, k=self.search_top_matches)
        self.logger.info("search_similar", status="retrieved", query=query, matches=len(matches))
        relevant_matches = [match for match, score in matches if score < self.score_threshold]
        self.logger.info("search_similar", status="finished", query=query, relevant_matches=len(relevant_matches))
        return relevant_matches

    def load_local_resources(self, local_resources_path: Path | str, extension: str = "md") -> None:
        self.logger.info(
            "load_local_resources",
            status="started",
            local_resources_path=local_resources_path,
            extension=extension,
        )
        if extension not in self.ALLOWED_EXTENSIONS:
            message = f"Extension '{extension}' not allowed. Must be one of {self.ALLOWED_EXTENSIONS}."
            self.logger.error(
                "load_local_resources",
                status="failed",
                message=message,
            )
            raise NotImplementedError(message)

        resources = DirectoryLoader(str(local_resources_path), glob=f"*.{extension}", show_progress=True).load()
        self._load_resources(resources)
        self.logger.info(
            "load_local_resources",
            status="finished",
        )

    def load_resource(self, resource: str) -> None:
        self.logger.info("load_resource", status="started")
        self._load_resources([Document(page_content=resource)])
        self.logger.info("load_resource", status="finished")

    def _load_resources(self, resources: list[Document]) -> None:
        self.logger.info("_load_resources", status="started", resources_count=len(resources))

        resource_chunks: list[Document] = self._chunk_resources(resources)
        chunks_to_add: list[Document] = list()
        for chunk in resource_chunks:
            document_from_db = self.client.get(where_document={"$contains": chunk.page_content})
            if document_from_db["ids"] and chunk.metadata in document_from_db["metadatas"]:
                self.logger.info(
                    "_load_resources",
                    status="skipped",
                    chunk_content=chunk.page_content[:50].replace("\n", " "),
                    reason="already exists in db",
                )
                continue
            else:
                chunks_to_add.append(chunk)

        if chunks_to_add:
            self._save_to_chroma(chunks_to_add)
        self.logger.info("_load_resources", status="finished", resources_count=len(resources))

    def _chunk_resources(self, resources: list[Document]) -> list[Document]:
        self.logger.info("_chunk_resources", status="started", resources_count=len(resources))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(resources)
        self.logger.info("_chunk_resources", status="finished", chunks_count=len(chunks))
        return chunks

    def _save_to_chroma(self, chunks: list[Document]) -> None:
        self.logger.info("_save_to_chroma", status="started", chunks_count=len(chunks))
        self.client.add_texts([chunk.page_content for chunk in chunks], metadatas=[chunk.metadata for chunk in chunks])
        self.logger.info("_save_to_chroma", status="finished", chunks_count=len(chunks))
