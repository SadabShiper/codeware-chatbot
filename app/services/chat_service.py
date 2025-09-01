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

    async def process_message(self, user_id: str, question: str) -> ChatResponse:
        """
        Process user message and let LLM determine if it should trigger a flow
        """
        # Letting LLM decide if this should trigger a flow
        flow_decision = await self._llm_flow_detection(question)
        
        if flow_decision.get("trigger_flow", False):
            flow_id = flow_decision.get("flow_id")
            if flow_id:
                logger.info(f"Flow triggered: {flow_id} for user: {user_id}")
                return ChatResponse(
                    answer="Your request has been forwarded to our service team. They will contact you shortly.",
                    triggered_flow=True,
                    flow_id=flow_id
                )
        
        # Using RAG with local model if no flow triggered
        return await self._generate_rag_response(question)

    async def _llm_flow_detection(self, question: str) -> dict:
        """
        Use LLM to determine if the question should trigger a specific flow
        """
        try:
            # Get all available flows for context
            available_flows = self.flow_service.get_all_flows()
            
            prompt = f"""
            Analyze the user's question and determine if it matches any of the available service flows.
            Available flows: {available_flows}
            
            User question: "{question}"
            
            Respond with JSON only in this format:
            {{
                "trigger_flow": true/false,
                "flow_id": "flow_id_string_if_applicable" OR null,
                "confidence": 0.0-1.0,
                "reason": "brief explanation"
            }}
            
            Only trigger a flow if the user is clearly requesting a specific service like package information, 
            new connection, bill payment, service request, or coverage check.
            """
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json"
            )
            
            # Parse LLM response
            decision = json.loads(response["message"]["content"])
            return decision
            
        except Exception as e:
            logger.error(f"Error in flow detection: {str(e)}")
            return {"trigger_flow": False, "flow_id": None, "confidence": 0.0, "reason": "error"}
    
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

            1. Use the context provided below to answer the user's question.

            2. If the context does not contain the answer, respond politely that you do not have that information.

            3. The user may ask questions in English, Bangla, or Banglish; respond in the same language.

            4. Provide concise, clear, and helpful answers suitable for a customer support setting.

            5. If the user is asking about specific services (packages, new connections, bill payments, 
               service requests, or coverage), consider whether this should be handled by a specialized flow.
               
            6. Include references to the sources from which you derived the answer whenever possible.
            
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
                answer="I'm having some trouble answering right now. Please try again later.",
                sources=[]
            )
