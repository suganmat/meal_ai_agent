"""
Meal Suggestion Agent for creating personalized meal recommendations.
"""
import logging
from typing import Dict, Any, Optional, List
from ..services.openrouter_client import OpenRouterClient
from ..services.perplexity_client import PerplexityClient
from ..services.neo4j_service import Neo4jService
from ..models.user_models import UserProfile
from ..models.meal_models import MealSuggestion, NutritionInfo, MealType
from ..utils.session_manager import SessionManager
from ..models.user_models import UserProfile, MedicalCondition
from langchain_core.tools import tool
from ..services.openrouter_client import OpenRouterClient
logger = logging.getLogger(__name__)


class MealSuggesterAgent:
    """Agent responsible for suggesting personalized meals."""
    
    def __init__(self, openrouter_client: OpenRouterClient, perplexity_client: PerplexityClient, neo4j_service: Neo4jService, session_manager: SessionManager):
        """
        Initialize the meal suggester agent.
        
        Args:
            OpenRouterClient: OpenRouter API client
            neo4j_service: Neo4j database service
            session_manager: Session manager instance
        """
        self.openrouter_client = openrouter_client
        self.perplexity_client = perplexity_client
        self.tools = [self.search_recipes_tool]
        self.session_manager = session_manager
        self.neo4j_service = neo4j_service
        
        # Meal suggestion prompt
        self.prompt = """
            You are a Meal Suggestion Agent that creates personalized meal recommendations based on user profiles.

            TASK: Suggest appropriate meals considering the specific user profile provided below.

            WORKFLOW:
            1. First ask: "What meal are you planning for - breakfast, lunch, or dinner?"
            2. Analyze the user profile provided and create appropriate meal suggestions
            3. Suggest ONE specific meal with complete details
            4. If user was previously dissatisfied, ensure the new suggestion is DIFFERENT from previous ones

            IMPORTANT: Use the user profile information intelligently to:
            - Consider any medical conditions and their dietary implications
            - Adjust recommendations based on age, BMI, and health status
            - Respect cuisine preferences and cultural considerations
            - Provide appropriate portion sizes and nutritional balance
            - Use the search_recipes_tool to find real, authentic recipes


            MEAL SUGGESTION FORMAT:
            **[Meal Name]**
            - **Preparation Time:** [X] minutes
            - **Nutrition (per serving):**
            [List the top Nutrients in the meal]
            - **Ingredients:**
            [List with quantities]
            - **Instructions:**
            [Step-by-step preparation instructions]

            GENERAL GUIDELINES:
            - Consider any medical conditions mentioned in the user profile
            - Use the cuisine preferences provided
            - Adjust portions based on age and BMI
            - Provide realistic nutrition values and practical cooking instructions
            - Be creative and authentic to the cuisine preferences

            AVAILABLE TOOLS:
            - search_recipes_tool: Search for real recipes using Perplexity API


            {user_profile_data}"""
    
    def process_message(self, message: str, session_id: str) -> str:
        """
        Process a user message and return meal suggestion.
        
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
            
            # Get user profile (from Neo4j for returning users or session for first-time users)
            user_profile = self._get_user_profile(session_id, session)
            if not user_profile:
                return "I need to collect your profile information first. Let me help you with that."
            
            # Get conversation history
            history = self.session_manager.get_conversation_history(session_id, limit=10)
            
            # Build personalized prompt with parameter injection
            personalized_prompt = self._build_personalized_prompt(user_profile)
            
            # Check if user was previously dissatisfied
            satisfaction = session.get("satisfaction", {})
            previous_suggestion = session.get("meal_suggestion", {}).get("suggestion")
            is_retry = satisfaction.get("level") == "not_satisfied" and satisfaction.get("wants_new_suggestion", False)
            
            # Build context about previous suggestions
            context_info = f"""
CURRENT MEAL SUGGESTION STATE: {session.get('meal_suggestion', {})}
PREVIOUS SUGGESTION: {previous_suggestion if previous_suggestion else 'None'}
USER DISSATISFIED: {is_retry}
"""
            
            if is_retry and previous_suggestion:
                context_info += f"""
IMPORTANT: User was dissatisfied with the previous suggestion. Provide a DIFFERENT meal suggestion.
Previous suggestion was: {previous_suggestion[:100]}...
Make sure to suggest something completely different in terms of cuisine, ingredients, or cooking style.
"""

            # Build messages for LLM - simplified format to avoid Perplexity API issues
            messages = [
                {"role": "system", "content": personalized_prompt + context_info},
                {"role": "user", "content": f"Current meal suggestion state: {session.get('meal_suggestion', {})}\nUser message: {message}"}
            ]
            
            # Get LLM response
            
            response = self.openrouter_client.chat_completion_with_tools(messages, tools=self.tools)
            agent_response = response["choices"][0]["message"]["content"]
            
            # Add messages to history
            self.session_manager.add_message_to_history(session_id, "user", message)
            self.session_manager.add_message_to_history(session_id, "assistant", agent_response)
            
            # Update session state
            self._update_session_state(session_id, message, agent_response)
            
            return agent_response
        
        except Exception as e:
            logger.error(f"Error in meal suggester: {e}")
            return "I'm sorry, I encountered an error while suggesting meals. Please try again."

    def _get_user_profile(self, session_id: str, session: Dict[str, Any]) -> Optional[UserProfile]:
        """
        Get user profile from Neo4j (returning users) or session (first-time users).
        
        Args:
            session_id: Session identifier
            session: Session data
            
        Returns:
            UserProfile object or None
        """
        # For returning users: get from Neo4j
        user_id = session.get("user_id")
        if user_id:
            user_profile = self.neo4j_service.get_user_profile(user_id)
            if user_profile:
                return user_profile
        
        # For first-time users: build from session data
        profile_data = self.session_manager.get_profile_collection(session_id)
        if profile_data.get("is_complete"):
            return self._create_user_profile_from_session(profile_data)
        
        return None
    
    def _build_personalized_prompt(self, user_profile: UserProfile) -> str:
        """
        Build a personalized prompt with parameter injection.
        
        Args:
            user_profile: User profile data
            
        Returns:
            Personalized prompt string
        """
        # Build user profile data section
        user_profile_data = self._build_user_profile_section(user_profile)
        
        
        # Inject parameters into the template
        personalized_prompt = self.prompt.format(
            user_profile_data=user_profile_data
        )
        
        return personalized_prompt

    def _build_user_profile_section(self, user_profile: UserProfile) -> str:
        """
        Build user profile section for the prompt.
        
        Args:
            user_profile: User profile data
            
        Returns:
            Formatted user profile string
        """
        profile_info = []
        
        profile_info.append(f"Name: {user_profile.name}")
        profile_info.append(f"Age: {user_profile.age} years")
        profile_info.append(f"Height: {user_profile.height} cm")
        profile_info.append(f"Weight: {user_profile.weight} kg")
        profile_info.append(f"BMI: {user_profile.bmi:.1f}")
        
        if user_profile.medical_conditions:
            conditions = [f"{cond.condition} ({cond.intensity})" for cond in user_profile.medical_conditions]
            profile_info.append(f"Medical Conditions: {', '.join(conditions)}")
        else:
            profile_info.append("Medical Conditions: None")
        
        if user_profile.primary_cuisine:
            profile_info.append(f"Primary Cuisine Preference: {user_profile.primary_cuisine}")
        
        if user_profile.secondary_cuisine:
            profile_info.append(f"Secondary Cuisine Preference: {user_profile.secondary_cuisine}")
        
        return "\n".join(profile_info)
    
    def _update_session_state(self, session_id: str, user_message: str, agent_response: str) -> None:
        """
        Update session state based on conversation.
        """
        try:
            # Update session with the new suggestion and clear previous satisfaction data
            self.session_manager.update_session(session_id, {
                "meal_suggestion": {
                    "suggestion": agent_response
                },
                "satisfaction": {}  # Clear previous satisfaction data
            })            
        except Exception as e:
            logger.error(f"Error updating session state: {e}")

    def _create_user_profile_from_session(self, profile_data: Dict[str, Any]) -> Optional[UserProfile]:
        """
        Create UserProfile object from session data (for first-time users).
        
        Args:
            profile_data: Profile data from session
            
        Returns:
            UserProfile object or None if invalid
        """
        try:
            # Convert medical conditions
            medical_conditions = []
            for condition_data in profile_data.get("medical_conditions", []):
                if isinstance(condition_data, dict) and "condition" in condition_data and "intensity" in condition_data:
                    medical_conditions.append(MedicalCondition(
                        condition=condition_data["condition"],
                        intensity=condition_data["intensity"]
        ))
            
            return UserProfile(
                name=profile_data["name"],
                age=profile_data["age"],
                height=profile_data.get("height"),
                weight=profile_data.get("weight"),
                medical_conditions=medical_conditions,
                primary_cuisine=profile_data["primary_cuisine"],
                secondary_cuisine=profile_data.get("secondary_cuisine")
            )
            
        except Exception as e:
            logger.error(f"Error creating user profile from session: {e}")
            return None
            
    @tool
    def search_recipes_tool(self, query: str, cuisine: str = None) -> str:
        """Search for recipes using Perplexity API."""
        return self.perplexity_client.search_recipes(query, cuisine)

    
    
