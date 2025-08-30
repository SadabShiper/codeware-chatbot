import logging
from typing import Optional
from app.models import ChatResponse
from app.services.flow_service import FlowService
from app.rag.retriever import RAGRetriever
import ollama  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.flow_service = FlowService()
        self.rag_retriever = RAGRetriever()
        self.model_name = model_name

        # Keywords for flow triggering
        self.trigger_keywords = {
            "packages": ["package", "pack", "plan", "price", "মূল্য", "প্যাকেজ", "প্যাক", "প্ল্যান"],
            "new_connection": ["new connection", "connection", "install", "setup", "নতুন সংযোগ", "কানেকশন", "ইনস্টল"],
            "bill_pay": ["bill", "payment", "pay", "বিল", "পেমেন্ট", "পরিশোধ"],
            "service_request": ["service", "problem", "issue", "help", "সেবা", "সমস্যা", "হেল্প"],
            "coverage": ["coverage", "area", "location", "কভারেজ", "এলাকা", "লোকেশন"]
        }

    async def process_message(self, user_id: str, question: str) -> ChatResponse:
        """
        Process user message and determine if it should trigger a flow or use RAG
        """
        # Step 1: Check for flow triggers
        flow_trigger = self._check_flow_trigger(question)
        if flow_trigger:
            logger.info(f"Flow triggered: {flow_trigger} for user: {user_id}")
            return ChatResponse(
                answer="Your request has been forwarded to our service team. They will contact you shortly.",
                triggered_flow=True,
                flow_id=flow_trigger
            )
        
        # Step 2: Use RAG with local model
        return await self._generate_rag_response(question)

    def _check_flow_trigger(self, question: str) -> Optional[str]:
        """
        Check if the user's question matches any predefined flow triggers
        """
        question_lower = question.lower()
        
        for flow, keywords in self.trigger_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                flow_ids = {
                    "packages": "679e564098ea05fc9dd74968_ad3734fab0d51f1a",
                    "new_connection": "679e564098ea05fc9dd74964_5b703bf48a2b99f0",
                    "bill_pay": "679e564098ea05fc9dd7496c_1a827ff9bcbc67a2",
                    "service_request": "679e564098ea05fc9dd7497a_4ca15b5e495f38cd",
                    "coverage": "679e564098ea05fc9dd7498a_9f13bae58a3a5d98"
                }
                return flow_ids[flow]
        return None

    async def _generate_rag_response(self, question: str) -> ChatResponse:
        """
        Generate response using RAG with Ollama local model
        """
        try:
            # Retrieve relevant information from knowledge base
            retrieved_info = self.rag_retriever.retrieve_relevant_info(question, top_k=3)
            context = self.rag_retriever.format_context(retrieved_info)
            
            sources = [info.get('metadata', {}).get('type', 'unknown') for info in retrieved_info]
            
            prompt = f"""
            You are a helpful customer support assistant for iDesk360, an internet service provider.
            Instructions:

            1. Use the context provided below to answer the user’s question.

            2. If the context does not contain the answer, respond politely that you do not have that information.

            3. The user may ask questions in English, Bangla, or Banglish; respond in the same language.

            4. Provide concise, clear, and helpful answers suitable for a customer support setting.

            5. Include references to the sources from which you derived the answer whenever possible.
            
            Context:
            {context}
            
            Question: {question}
            """

            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response["message"]["content"] if "message" in response else "Sorry, I could not generate a response."

            return ChatResponse(
                answer=answer,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return ChatResponse(
                answer="I’m having some trouble answering right now. Please try again later.",
                sources=[]
            )