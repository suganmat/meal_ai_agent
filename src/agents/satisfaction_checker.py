"""
Satisfaction Checker Agent for ensuring user satisfaction with meal suggestions.
"""
import logging
from typing import Dict, Any, Optional, List
from ..services.openrouter_client import OpenRouterClient
from ..utils.session_manager import SessionManager

logger = logging.getLogger(__name__)


class SatisfactionCheckerAgent:
    """Agent responsible for checking user satisfaction and handling feedback."""
    
    def __init__(self, openrouter_client: OpenRouterClient, session_manager: SessionManager):
        """
        Initialize the satisfaction checker agent.
        
        Args:
            openrouter_client: OpenRouter API client
            session_manager: Session manager instance
        """
        self.openrouter_client = openrouter_client
        self.session_manager = session_manager
        
        # Satisfaction checker prompt
        self.prompt = """
You are a Satisfaction Checker Agent that ensures users are happy with meal suggestions.

TASK: Check user satisfaction using natural language understanding and handle their response appropriately.

WORKFLOW:
1. Analyze the user's response using natural language understanding to determine satisfaction
2. Provide appropriate response based on their sentiment
3. Ask follow-up questions when needed

RESPONSE LOGIC:
- If SATISFIED (user expresses positive sentiment, approval, or acceptance): 
  * Respond with encouraging message
  * Provide a helpful cooking tip
  * End with "Enjoy your meal! Feel free to ask for more suggestions anytime."
  * User will return to normal chat for further conversation
  
- If NOT SATISFIED (user expresses negative sentiment, rejection, or dissatisfaction):
  * Acknowledge their dissatisfaction empathetically
  * Ask: "Would you like me to suggest a different meal?"
  * Wait for their response to determine next steps

- If USER WANTS NEW SUGGESTION (after being dissatisfied):
  * Confirm you'll provide a different suggestion
  * User will go back to meal suggestion node for new recommendation

- If USER DOESN'T WANT NEW SUGGESTION (after being dissatisfied):
  * Accept their decision gracefully
  * User will return to normal chat

EXAMPLES:
SATISFIED responses:
- "Wonderful! Here's a tip: marinate the chicken for at least 30 minutes for best flavor. Enjoy your meal!"

NOT SATISFIED responses:
- "I understand you're not satisfied with this suggestion. Would you like me to suggest a different meal?"

Always be supportive and solution-focused. Use natural language understanding rather than keyword matching to detect sentiment.
"""
    
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
            # Get session data
            session = self.session_manager.get_session(session_id)
            if not session:
                return "I'm sorry, your session has expired. Please start over."
            
            # Get conversation history
            history = self.session_manager.get_conversation_history(session_id, limit=10)
            
            # Build messages for LLM - simplified format to avoid OpenRouter API issues
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"Current meal suggestion: {session.get('meal_suggestion', {})}\nCurrent satisfaction state: {session.get('satisfaction', {})}\nUser message: {message}"}
            ]
            
            # Get LLM response
            
            response = self.openrouter_client.chat_completion(messages)
            agent_response = response["choices"][0]["message"]["content"]
            
            # Add messages to history
            self.session_manager.add_message_to_history(session_id, "user", message)
            self.session_manager.add_message_to_history(session_id, "assistant", agent_response)
            
            # Update session state
            self._update_session_state(session_id, message, agent_response)
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in satisfaction checker: {e}")
            return "I'm sorry, I encountered an error. Please try again."
    
    def _update_session_state(self, session_id: str, user_message: str, agent_response: str) -> None:
        """
        Update session state based on conversation.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            agent_response: Agent's response
        """
        try:
            # Determine satisfaction level
            satisfaction_level = self._determine_satisfaction(user_message)
            
            # Update session with satisfaction data
            self.session_manager.update_session(session_id, {
                "satisfaction": {
                    "level": satisfaction_level,
                    "feedback": user_message,
                    "response": agent_response
                }
            })
            
            # Update session state based on satisfaction
            if satisfaction_level == "satisfied":
                self.session_manager.update_session_state(session_id, "normal_chat")  # ✅ Go back to normal chat
            elif satisfaction_level == "not_satisfied":
                # Check if user wants a new suggestion
                wants_new_suggestion = self._determine_wants_new_suggestion(user_message)
                self.session_manager.update_session(session_id, {
                    "satisfaction": {
                        "level": satisfaction_level,
                        "wants_new_suggestion": wants_new_suggestion
                    }
                })
                # Don't change session state here - let routing logic handle it
            
        except Exception as e:
            logger.error(f"Error updating session state: {e}")
    
    def _determine_satisfaction(self, message: str) -> str:
        """
        Determine user satisfaction level from message using LLM-based sentiment analysis.
        
        Args:
            message: User's message
            
        Returns:
            Satisfaction level: "satisfied", "not_satisfied", or "neutral"
        """
        try:
            sentiment_prompt = f"""
Analyze the user's response to determine their satisfaction level with a meal suggestion.

User message: "{message}"

Determine if the user is:
- SATISFIED: Expresses positive sentiment, approval, acceptance, or enthusiasm
- NOT_SATISFIED: Expresses negative sentiment, rejection, dissatisfaction, or concerns
- NEUTRAL: Unclear or mixed sentiment

Respond with ONLY one word: SATISFIED, NOT_SATISFIED, or NEUTRAL
"""
            
            
            response = self.openrouter_client.chat_completion([
                {"role": "user", "content": sentiment_prompt}
            ])
            
            sentiment = response["choices"][0]["message"]["content"].strip().upper()
            
            if "SATISFIED" in sentiment:
                return "satisfied"
            elif "NOT_SATISFIED" in sentiment:
                return "not_satisfied"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Fallback to neutral if LLM analysis fails
            return "neutral"
    
    def _determine_wants_new_suggestion(self, message: str) -> bool:
        """
        Determine if user wants a new meal suggestion using LLM analysis.
        
        Args:
            message: User's message
            
        Returns:
            True if user wants new suggestion, False otherwise
        """
        try:
            wants_new_prompt = f"""
Analyze the user's response to determine if they want a new meal suggestion.

User message: "{message}"

Determine if the user wants:
- NEW SUGGESTION: Expresses desire for different meal, alternative, or new recommendation
- NO NEW SUGGESTION: Expresses they don't want another suggestion, are done, or want to move on

Respond with ONLY one word: YES or NO
"""
            
            
            response = self.openrouter_client.chat_completion([
                {"role": "user", "content": wants_new_prompt}
            ])
            
            wants_new = response["choices"][0]["message"]["content"].strip().upper()
            return "YES" in wants_new
            
        except Exception as e:
            logger.error(f"Error determining wants new suggestion: {e}")
            # Fallback to True if LLM analysis fails (safer to offer new suggestion)
            return True
    
    def get_cooking_tip(self, meal_type: str, cuisine: str) -> str:
        """
        Get a cooking tip based on meal type and cuisine.
        
        Args:
            meal_type: Type of meal
            cuisine: Type of cuisine
            
        Returns:
            Cooking tip string
        """
        tips = {
            "indian": {
                "breakfast": "Pro tip: Toast the spices lightly before adding to enhance their flavor!",
                "lunch": "Pro tip: Let the chicken marinate for at least 30 minutes for maximum flavor absorption!",
                "dinner": "Pro tip: Soak lentils overnight for faster cooking and better texture!"
            },
            "italian": {
                "breakfast": "Pro tip: Use fresh herbs for the best flavor in your morning dishes!",
                "lunch": "Pro tip: Cook pasta al dente and finish it in the sauce for perfect texture!",
                "dinner": "Pro tip: Let your sauce simmer slowly to develop rich, deep flavors!"
            },
            "chinese": {
                "breakfast": "Pro tip: High heat and quick cooking preserve the crisp texture of vegetables!",
                "lunch": "Pro tip: Marinate proteins briefly before stir-frying for tender results!",
                "dinner": "Pro tip: Add aromatics like ginger and garlic at the end to preserve their fragrance!"
            }
        }
        
        cuisine_tips = tips.get(cuisine.lower(), tips["indian"])
        return cuisine_tips.get(meal_type.lower(), "Pro tip: Taste and adjust seasoning as you cook!")
    
    def collect_feedback(self, session_id: str, meal_suggestion: str) -> Dict[str, Any]:
        """
        Collect detailed feedback from user.
        
        Args:
            session_id: Session identifier
            meal_suggestion: The meal suggestion to get feedback on
            
        Returns:
            Feedback data
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {}
        
        satisfaction_data = session.get("satisfaction", {})
        
        return {
            "meal_suggestion": meal_suggestion,
            "satisfaction_level": satisfaction_data.get("level", "neutral"),
            "feedback_text": satisfaction_data.get("feedback", ""),
            "timestamp": session.get("last_accessed"),
            "user_id": session.get("user_id")
        }
    
    def generate_alternative_suggestion(self, feedback: str, user_profile: Dict[str, Any]) -> str:
        """
        Generate alternative meal suggestion based on feedback.
        
        Args:
            feedback: User's feedback
            user_profile: User's profile information
            
        Returns:
            Alternative suggestion
        """
        # This would integrate with the meal suggester to create alternatives
        # For now, return a generic response
        return f"Based on your feedback about '{feedback}', I'll suggest something different that better matches your preferences. Let me find you a better option!"
    
    def is_conversation_complete(self, session_id: str) -> bool:
        """
        Check if the conversation is complete (user is satisfied).
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if conversation is complete, False otherwise
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return False
        
        satisfaction_data = session.get("satisfaction", {})
        return satisfaction_data.get("level") == "satisfied"
