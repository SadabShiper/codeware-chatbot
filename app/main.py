from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from app.services.chat_service import ChatService
from app.models import ChatRequest, ChatResponse

load_dotenv()

app = FastAPI(title="Multilingual RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


chat_service = ChatService(model_name="llama3.2:3b")

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/")
async def root():
    return {"message": "Multilingual RAG Chatbot API"}

@app.get("/health")
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", message="Service is running")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await chat_service.process_message(
            user_id=request.user_id,
            question=request.question
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/ingest")
async def ingest_data():
    try:
        chat_service.rag_retriever._initialize_knowledge_base()
        return {"status": "success", "message": "Data ingestion completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
