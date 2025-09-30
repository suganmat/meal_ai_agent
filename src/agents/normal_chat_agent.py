"""
Normal Chat Agent for handling general conversations.
"""
import logging
from typing import Dict, Any, Optional
from ..services.perplexity_client import PerplexityClient
from ..utils.session_manager import SessionManager
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


class NormalChatAgent:
    """Agent responsible for handling general conversations."""
    
    def __init__(self, perplexity_client: PerplexityClient, session_manager: SessionManager):
        """
        Initialize the normal chat agent.
        
        Args:
            perplexity_client: Perplexity API client
            session_manager: Session manager instance
        """
        self.perplexity_client = perplexity_client
        self.session_manager = session_manager
        
        # General chat prompt
        self.prompt = """
        You are a helpful AI assistant. Be friendly, conversational, and do a chat like ChatGPT. 
        Try to have conversation with the user for all their inputs."""
    
    def process_message(self, message: str, session_id: str) -> str:
        """
        Process a user message and return appropriate response.
        
        Args:
            message: User's message
            session_id: Session identifier
            
        Returns:
            Agent's response
        """
        try:
            # Try OpenRouter API first
            try:
                chat_messages = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": message}
                ]
                
                
                response = self.perplexity_client.chat_completion(chat_messages)
                agent_response = response["choices"][0]["message"]["content"]
                
                # Check if we got a rate limit response and provide better fallback
                if "currently experiencing high demand" in agent_response or "rate limit" in agent_response.lower():
                    logger.warning("Received rate limit response, using fallback")
                    user_message = message.lower()
                    if any(greeting in user_message for greeting in ["hello", "hi", "hey"]):
                        agent_response = "Hello! I'm here to help you with meal recommendations and general conversation. How can I assist you today?"
                    elif any(question in user_message for question in ["how are you", "how are you doing"]):
                        agent_response = "I'm doing well, thank you for asking! I'm ready to help you with meal suggestions or any other questions you might have."
                    elif any(weather in user_message for weather in ["weather", "temperature", "rain", "sunny"]):
                        agent_response = "I don't have access to current weather information, but I'd be happy to help you with meal recommendations or other questions!"
                    elif any(intro in user_message for intro in ["tell me about", "who are you", "what are you"]):
                        agent_response = "I'm an AI assistant specialized in meal recommendations and general conversation. I can help you find perfect meals based on your preferences, dietary needs, and taste preferences. What would you like to know?"
                    else:
                        agent_response = "I'm here to help! Feel free to ask me about meal recommendations or anything else."
                
            except Exception as api_error:
                logger.warning(f"API error in normal chat, using fallback: {api_error}")
                # Simple fallback responses
                user_message = message.lower()
                if any(greeting in user_message for greeting in ["hello", "hi", "hey"]):
                    agent_response = "Hello! I'm here to help you with meal recommendations and general conversation. How can I assist you today?"
                elif any(question in user_message for question in ["how are you", "how are you doing"]):
                    agent_response = "I'm doing well, thank you for asking! I'm ready to help you with meal suggestions or any other questions you might have."
                elif any(weather in user_message for weather in ["weather", "temperature", "rain", "sunny"]):
                    agent_response = "I don't have access to current weather information, but I'd be happy to help you with meal recommendations or other questions!"
                elif any(intro in user_message for intro in ["tell me about", "who are you", "what are you"]):
                    agent_response = "I'm an AI assistant specialized in meal recommendations and general conversation. I can help you find perfect meals based on your preferences, dietary needs, and taste preferences. What would you like to know?"
                else:
                    agent_response = "I'm currently experiencing some technical difficulties, but I'm here to help! Feel free to ask me about meal recommendations or anything else."
            
            # Add to session history
            self.session_manager.add_message_to_history(session_id, "user", message)
            self.session_manager.add_message_to_history(session_id, "assistant", agent_response)
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in normal chat: {e}")
            return "I'm sorry, I encountered an error. Please try again."
    
    def process_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the normal chat node.
        
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
            
            # Process with normal chat
            agent_response = self.process_message(user_message, session_id)
            
            # Update state
            state["messages"] = add_messages(state["messages"], [AIMessage(content=agent_response)])
            state["current_state"] = "normal_chat"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in normal chat node: {e}")
            state["messages"] = add_messages(state["messages"], [
                AIMessage(content="I'm sorry, I encountered an error. Please try again.")
            ])
            return state
