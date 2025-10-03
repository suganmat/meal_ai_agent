"""
Intent Detection Agent for analyzing user input to determine conversation intent.
"""
import logging
from typing import Dict, Any, Optional
from ..services.openrouter_client import OpenRouterClient
from ..utils.session_manager import SessionManager
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


class IntentDetectionAgent:
    """Agent responsible for detecting user intent from messages."""
    
    def __init__(self, openrouter_client: OpenRouterClient, session_manager: SessionManager):
        """
        Initialize the intent detection agent.
        
        Args:
            openrouter_client: OpenRouter API client
            session_manager: Session manager instance
        """
        self.openrouter_client = openrouter_client
        self.session_manager = session_manager
        
        # Intent detection prompt
        self.prompt = """
        Analyze this user message and determine the intent:
        User message: "{user_message}"
        
        Respond with ONLY one of these options:
        - "meal_request" if the user is asking for meal suggestion, recommendations, food suggestions, or anything related to meals/food
        - "normal_chat" for any other conversation (greetings, general questions, etc.)
        
        Examples:
        - "I want meal recommendations" -> meal_request
        - "What should I eat for dinner?" -> meal_request
        - "suggest a meal" -> meal_request
        - "I need food suggestions" -> meal_request
        - "Hello, how are you?" -> normal_chat
        - "What's the weather like?" -> normal_chat
        - "Tell me a joke" -> normal_chat
        - "Thank you" -> normal_chat
        
        IMPORTANT: Respond with ONLY the word "meal_request" or "normal_chat", nothing else.
        """
    
    def process_message(self, message: str, session_id: str) -> str:
        """
        Process a user message and detect intent.
        
        Args:
            message: User's message
            session_id: Session identifier
            
        Returns:
            Detected intent ("meal_request" or "normal_chat")
        """
        try:
            # Use OpenRouter to detect intent
            intent_prompt = self.prompt.format(user_message=message)
            
            
            response = self.openrouter_client.chat_completion([
                {"role": "user", "content": intent_prompt}
            ])
            
            raw_response = response["choices"][0]["message"]["content"].strip().lower()
            
            # Extract intent from potentially verbose response
            if "meal_request" in raw_response:
                intent = "meal_request"
            elif "normal_chat" in raw_response:
                intent = "normal_chat"
            else:
                intent = "normal_chat"  # Default fallback
            
            return intent
            
        except Exception as e:
            logger.error(f"Error in intent detection: {e}")
            return "normal_chat"  # Default fallback
    
    def process_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the intent detection node.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            session_id = state["session_id"]
            messages = state["messages"]
            
            # Get the latest user message
            if messages:
                latest_message = messages[-1]
                if isinstance(latest_message, HumanMessage):
                    user_message = latest_message.content
                else:
                    user_message = str(latest_message)
            else:
                user_message = "Hello!"
            
            # Detect intent
            intent = self.process_message(user_message, session_id)
            
            # Update state
            state["intent"] = intent
            state["current_state"] = "intent_detection"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in intent detection node: {e}")
            state["intent"] = "normal_chat"
            state["current_state"] = "intent_detection"
            return state
