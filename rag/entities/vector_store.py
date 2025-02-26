from pydantic import BaseModel

class VectorStoreQueryParams(BaseModel):
    query: str
    max_results: int