from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    triggered_flow: Optional[bool] = False
    flow_id: Optional[str] = None

class FlowTriggerRequest(BaseModel):
    user_id: str
    trigger_id: str

class FlowTriggerResponse(BaseModel):
    success: bool
    message: str
    next_step: Optional[str] = None