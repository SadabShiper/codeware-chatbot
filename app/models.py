from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    triggered_flow: Optional[bool] = False
    flow_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

class IngestResponse(BaseModel):
    status: str
    message: str