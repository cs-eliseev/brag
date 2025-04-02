from pathlib import Path
from typing import List, Tuple
from rag.drivers.databases.faiss_db import FaissDB
from rag.drivers.embeddings.embedding import EmbeddingWrapper

class VectorDatabaseService:
    def __init__(self, embedding: EmbeddingWrapper):
        self.embedding = embedding

    def get_vector_databases(self) -> List[Tuple[str, FaissDB]]:
        database_dir = Path("databases")
        vector_dbs = []

        for db_path in database_dir.iterdir():
            if db_path.is_dir() and (db_path / "index.faiss").exists():
                db_name = db_path.name
                db = FaissDB(db_path=db_path, embeddings=self.embedding)
                vector_dbs.append((db_name, db))

        return vector_dbs 