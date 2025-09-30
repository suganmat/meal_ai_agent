"""
Meal Suggestion Agent for creating personalized meal recommendations.
"""
import logging
from typing import Dict, Any, Optional, List
from ..services.perplexity_client import PerplexityClient
from ..services.neo4j_service import Neo4jService
from ..models.user_models import UserProfile
from ..models.meal_models import MealSuggestion, NutritionInfo, MealType
from ..utils.session_manager import SessionManager
from ..models.user_models import UserProfile, MedicalCondition
from langchain_core.tools import tool
from ..services.perplexity_client import PerplexityClient
logger = logging.getLogger(__name__)


class MealSuggesterAgent:
    """Agent responsible for suggesting personalized meals."""
    
    def __init__(self, perplexity_client: PerplexityClient, neo4j_service: Neo4jService, session_manager: SessionManager):
        """
        Initialize the meal suggester agent.
        
        Args:
            perplexity_client: Perplexity API client
            neo4j_service: Neo4j database service
            session_manager: Session manager instance
        """
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


            {user_profile_data}

            {specific_instructions}"""
    
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
            
            response = self.perplexity_client.chat_completion_with_tools(messages, tools=self.tools)
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
        
        # Build specific instructions section
        specific_instructions = self._build_specific_instructions(user_profile)
        
        # Inject parameters into the template
        personalized_prompt = self.prompt.format(
            user_profile_data=user_profile_data,
            specific_instructions=specific_instructions
        )
        
        return personalized_prompt

    def _build_user_profile_section(self, user_profile: UserProfile) -> str:
        """
        Build the user profile data section for the prompt.
        
        Args:
            user_profile: User profile data
            
        Returns:
            Formatted user profile section
        """
        profile_section = """
            USER PROFILE DATA:
            - Name: {name}
            - Age: {age} years old
            - Primary Cuisine Preference: {primary_cuisine}"""

        if user_profile.secondary_cuisine:
            profile_section += "\n- Secondary Cuisine Preference: {secondary_cuisine}"
        
        if user_profile.height and user_profile.weight:
            profile_section += """
            - Height: {height} cm
            - Weight: {weight} kg
            - BMI: {bmi} ({bmi_category})"""
        
        if user_profile.medical_conditions:
            conditions = [f"{mc.condition} ({mc.intensity})" for mc in user_profile.medical_conditions]
            profile_section += "\n- Medical Conditions: {medical_conditions}"
            
            # Format the conditions string
            conditions_str = ', '.join(conditions)
        else:
            conditions_str = "None"
        
        # Format the profile section with actual data
        formatted_section = profile_section.format(
            name=user_profile.name,
            age=user_profile.age,
            primary_cuisine=user_profile.primary_cuisine,
            secondary_cuisine=user_profile.secondary_cuisine or "Not specified",
            height=user_profile.height or "Not provided",
            weight=user_profile.weight or "Not provided",
            bmi=user_profile.bmi or "Cannot calculate",
            bmi_category=user_profile.bmi_category or "Unknown",
            medical_conditions=conditions_str
        )
        
        return formatted_section

    def _build_specific_instructions(self, user_profile: UserProfile) -> str:
        """
        Build specific instructions based on user profile.
        
        Args:
            user_profile: User profile data
            
        Returns:
            Specific instructions string
        """
        instructions = []
        
        # Medical condition considerations
        if user_profile.medical_conditions:
            for condition in user_profile.medical_conditions:
                if condition.condition.lower() in ['diabetes', 'diabetic']:
                    instructions.append("- Avoid high-sugar ingredients and focus on low-glycemic index foods")
                elif condition.condition.lower() in ['hypertension', 'high blood pressure']:
                    instructions.append("- Minimize sodium content and avoid processed foods")
                elif condition.condition.lower() in ['heart disease', 'cardiac']:
                    instructions.append("- Focus on heart-healthy ingredients like omega-3 rich foods")
                elif condition.condition.lower() in ['obesity', 'weight management']:
                    instructions.append("- Suggest portion-controlled meals with balanced macronutrients")
        
        # BMI-based considerations
        if user_profile.bmi:
            if user_profile.bmi < 18.5:
                instructions.append("- Suggest nutrient-dense meals to support healthy weight gain")
            elif user_profile.bmi > 30:
                instructions.append("- Focus on lower-calorie, high-fiber options for weight management")
        
        # Age-based considerations
        if user_profile.age < 18:
            instructions.append("- Ensure meals support healthy growth and development")
        elif user_profile.age > 65:
            instructions.append("- Consider easier-to-digest options and nutrient absorption")
        
        # Cuisine-specific instructions
        if user_profile.primary_cuisine:
            instructions.append(f"- Prioritize authentic {user_profile.primary_cuisine} cuisine flavors and techniques")
        
        if not instructions:
            instructions.append("- Focus on balanced nutrition and appealing flavors")
        
        return "\n".join(instructions)

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
            self.session_manager.update_session_state(session_id, "satisfaction_check")
            
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

    
    
