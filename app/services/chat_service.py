import logging
from typing import Dict, Any, Optional
from app.models import ChatResponse
from app.services.flow_service import FlowService
from app.rag.retriever import RAGRetriever
import google.generativeai as genai
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.flow_service = FlowService()
        self.rag_retriever = RAGRetriever()
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
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
        
        # Step 2: Use RAG with Gemini for general questions
        return await self._generate_rag_response(question)

    def _check_flow_trigger(self, question: str) -> Optional[str]:
        """
        Check if the user's question matches any predefined flow triggers
        """
        question_lower = question.lower()
        
        # Check for package-related queries
        if any(keyword in question_lower for keyword in self.trigger_keywords["packages"]):
            return "679e564098ea05fc9dd74968_ad3734fab0d51f1a"  # Packages flow ID
        
        # Check for new connection queries
        if any(keyword in question_lower for keyword in self.trigger_keywords["new_connection"]):
            return "679e564098ea05fc9dd74964_5b703bf48a2b99f0"  # New Connection flow ID
        
        # Check for bill payment queries
        if any(keyword in question_lower for keyword in self.trigger_keywords["bill_pay"]):
            return "679e564098ea05fc9dd7496c_1a827ff9bcbc67a2"  # Bill Pay flow ID
        
        # Check for service requests
        if any(keyword in question_lower for keyword in self.trigger_keywords["service_request"]):
            return "679e564098ea05fc9dd7497a_4ca15b5e495f38cd"  # Service Request flow ID
        
        # Check for coverage queries
        if any(keyword in question_lower for keyword in self.trigger_keywords["coverage"]):
            return "679e564098ea05fc9dd7498a_9f13bae58a3a5d98"  # Coverage flow ID
        
        return None

    async def _generate_rag_response(self, question: str) -> ChatResponse:
        """
        Generate response using RAG with Gemini
        """
        try:
            # Retrieve relevant information from knowledge base
            retrieved_info = self.rag_retriever.retrieve_relevant_info(question, top_k=3)
            context = self.rag_retriever.format_context(retrieved_info)
            
            # Extract source information
            sources = [info.get('metadata', {}).get('type', 'unknown') for info in retrieved_info]
            
            # Create a prompt for Gemini with context
            prompt = f"""
            You are a helpful customer support assistant for iDesk360, an internet service provider.
            Use the following context information to answer the user's question. If the context doesn't
            contain the answer, respond politely that you don't have that information.
            
            Context:
            {context}
            
            Question: {question}
            
            Please provide a clear, concise, and helpful answer based on the context provided.
            If you need to ask for more information, do so politely.
            """
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            return ChatResponse(
                answer=response.text,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return ChatResponse(
                answer="I apologize, but I'm experiencing technical difficulties. Please try again later or contact our support team directly.",
                sources=[]
            )