import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

class FlowService:
    def __init__(self):
        self.flows = self._load_flows()
    
    def _load_flows(self) -> List[Dict[str, Any]]:
        """
        Load flow definitions from JSON file
        """
        try:
            flow_file = Path(__file__).parent.parent / "data" / "codeware_bot_flow.json"
            with open(flow_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading flows: {str(e)}")
            return []
    
    def get_flow_by_id(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get flow definition by ID
        """
        for flow in self.flows:
            if flow.get("id") == flow_id:
                return flow
        return None
    
    def get_all_flows(self) -> List[Dict[str, Any]]:
        """
        Return all available flows for LLM context
        """
        # Return simplified flow information for LLM context
        simplified_flows = []
        for flow in self.flows:
            simplified_flows.append({
                "id": flow.get("id"),
                "name": flow.get("name", "Unnamed Flow"),
                "description": flow.get("description", ""),
                "keywords": flow.get("keywords", []),
                "purpose": flow.get("purpose", "")
            })
        return simplified_flows
    
    def find_relevant_flow(self, user_intent: str) -> Optional[Dict[str, Any]]:
        """
        Find the most relevant flow based on user intent (can be used by LLM)
        """
        # This method can be enhanced with semantic matching if needed
        for flow in self.flows:
            if any(keyword.lower() in user_intent.lower() for keyword in flow.get("keywords", [])):
                return flow
        return None
