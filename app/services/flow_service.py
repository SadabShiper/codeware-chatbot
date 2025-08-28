import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class FlowService:
    def __init__(self):
        self.flows = self._load_flows()
    
    def _load_flows(self) -> Dict[str, Any]:
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
    
    def get_flow_by_keyword(self, keyword: str) -> Optional[Dict[str, Any]]:
        """
        Get flow that matches the given keyword
        """
        keyword_lower = keyword.lower()
        for flow in self.flows:
            flow_keywords = flow.get("keywords", [])
            if any(kw.lower() in keyword_lower for kw in flow_keywords):
                return flow
        return None