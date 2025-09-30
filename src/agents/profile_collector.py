"""
Profile Collection Agent for gathering user information.
"""
import logging
from typing import Dict, Any, Optional, List
from ..services.perplexity_client import PerplexityClient
from ..services.neo4j_service import Neo4jService
from ..models.user_models import UserProfile, MedicalCondition
from ..utils.session_manager import SessionManager
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


class ProfileCollectorAgent:
    """Agent responsible for collecting user profile information."""
    
    def __init__(self, perplexity_client: PerplexityClient, neo4j_service: Neo4jService, session_manager: SessionManager):
        """
        Initialize the profile collector agent.
        
        Args:
            perplexity_client: Perplexity API client
            neo4j_service: Neo4j database service
            session_manager: Session manager instance
        """
        self.perplexity_client = perplexity_client
        self.neo4j_service = neo4j_service
        self.session_manager = session_manager
        
        # Single comprehensive prompt that handles everything
        self.prompt = """
You are a Profile Collection Agent for a meal recommendation system. Your goal is to collect complete user profile information through friendly conversation.

TASK: Handle name extraction, conversation, and profile data extraction in a single response.

REQUIRED INFORMATION FOR NEW USERS:
- Basic Details: name, age, height (optional), weight (optional)
- Medical Conditions: condition name and intensity (mild/moderate/severe)
- Cuisine Preferences: primary cuisine, secondary cuisine (optional)

CONVERSATION STYLE:
- Be conversational and friendly, not robotic
- Accept information in any order
- If user gives partial info, acknowledge and ask for missing details
- Use natural follow-up questions
- For returning users, acknowledge their existing preferences

RESPONSE FORMAT:
Respond with a JSON object followed by your conversational response. Use this exact format:

```json
{
    "extracted_data": {
        "name": "extracted name or null",
        "age": "extracted age number or null", 
        "height": "extracted height number or null",
        "weight": "extracted weight number or null",
        "medical_conditions": [{"condition": "condition_name", "intensity": "mild/moderate/severe"}],
        "primary_cuisine": "extracted cuisine or null",
        "secondary_cuisine": "extracted cuisine or null"
    },
    "conversation_response": "Your conversational response here"
}
```

EXAMPLES:
- If no name: "Hi! I'm here to help you find perfect meals. What's your name?"
- If has name but no age: "Nice to meet you, John! What's your age?"
- If collecting preferences: "What type of cuisine do you usually enjoy? Italian, Indian, Chinese, etc.?"
- If profile complete: "Perfect! I've saved your profile. Now let's find you a great meal!"

Only include fields in extracted_data that are clearly mentioned in the user's message."""
    
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
            
            # Get conversation history and profile data
            history = self.session_manager.get_conversation_history(session_id, limit=10)
            profile_data = session.get("profile_collection", {})
            
            # Build context for the single prompt
            context_info = f"""
CURRENT STATE:
- Profile collection progress: {profile_data}
- Has name: {bool(profile_data.get('name'))}
- Current session state: {session.get('current_state', 'initial')}
"""
            
            # Single API call using the comprehensive prompt
            
            response = self.perplexity_client.chat_completion([
                {"role": "system", "content": self.prompt + context_info},
                {"role": "user", "content": message}
            ])
            
            agent_response = response["choices"][0]["message"]["content"]
            
            # Extract and process the structured data from the response
            self._process_combined_response(session_id, agent_response, message)
            
            # Add messages to history
            self.session_manager.add_message_to_history(session_id, "user", message)
            self.session_manager.add_message_to_history(session_id, "assistant", agent_response)
            
            return agent_response
            
        except Exception as e:
            logger.error(f"Error in profile collector: {e}")
            return "I'm sorry, I encountered an error. Please try again."
    
    
    def _process_combined_response(self, session_id: str, agent_response: str, user_message: str) -> None:
        """
        Process the combined response to extract data and handle special cases.
        
        Args:
            session_id: Session identifier
            agent_response: Agent's response
            user_message: Original user message
        """
        logger.info(f"Processing combined response for session {session_id}")
        logger.info(f"Response: {agent_response[:200]}...")
        logger.info(f"User message: {user_message}")
        
        # Get current session and profile data
        current_session = self.session_manager.get_session(session_id)
        current_profile = self.session_manager.get_profile_collection(session_id)
        logger.info(f"Current session exists: {current_session is not None}")
        logger.info(f"Current profile: {current_profile}")
        
        try:
            # Try to extract JSON data from the response
            import json
            import re
            
            # Look for JSON in the response
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, agent_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    extracted_data = json.loads(json_str)
                    
                    # Process the extracted data
                    if "extracted_data" in extracted_data:
                        data = extracted_data["extracted_data"]
                        
                        # Handle name extraction and database check
                        logger.info(f"Checking name extraction: data.get('name')={data.get('name')}, current_name={self.session_manager.get_profile_collection(session_id).get('name')}")
                        if data.get("name") and not self.session_manager.get_profile_collection(session_id).get("name"):
                            name = data["name"]
                            
                            # Check if user exists in database
                            if self.neo4j_service.check_user_exists(name):
                                # User exists, get their profile
                                existing_profile = self.get_existing_user_profile(name)
                                if existing_profile:
                                    # Update session with existing profile
                                    for field, value in existing_profile.items():
                                        self.session_manager.update_profile_collection(session_id, field, value)
                                    self.session_manager.update_session(session_id, {"user_name": name, "user_id": existing_profile.get("user_id")})
                                    self.session_manager.update_session_state(session_id, "meal_suggestion")
                                    return  # Skip further processing for returning users
                            else:
                                # New user, start collecting profile
                                self.session_manager.update_profile_collection(session_id, "name", name)
                                self.session_manager.update_session(session_id, {"user_name": name})
                        
                        # Update profile collection with extracted data
                        logger.info(f"Processing extracted data: {data}")
                        for field, value in data.items():
                            logger.info(f"Processing field {field} with value {value}")
                            if value is not None and value != "null" and value != "":
                                try:
                                    if field == "medical_conditions" and isinstance(value, list):
                                        # Handle medical conditions
                                        current_conditions = self.session_manager.get_profile_collection(session_id).get("medical_conditions", [])
                                        current_conditions.extend(value)
                                        success = self.session_manager.update_profile_collection(session_id, field, current_conditions)
                                        logger.info(f"Updated medical_conditions: {success}")
                                    else:
                                        # Handle other fields
                                        success = self.session_manager.update_profile_collection(session_id, field, value)
                                        logger.info(f"Updated {field} with {value}: {success}")
                                except Exception as e:
                                    logger.error(f"Error updating field {field}: {e}")
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from combined response: {e}")
            
            # Always update session state after data processing to prevent infinite loops
            profile_data = self.session_manager.get_profile_collection(session_id)
            if self._is_profile_complete(profile_data):
                # Create user profile in database if not already created
                if not self.session_manager.get_session(session_id).get("user_id"):
                    user_profile = self._create_user_profile_from_session(profile_data)
                    if user_profile:
                        user_id = self.neo4j_service.create_user_profile(user_profile)
                        self.session_manager.update_session(session_id, {"user_id": user_id})
                self.session_manager.update_session_state(session_id, "meal_suggestion")
            else:
                # Profile not complete, stay in profile collection
                self.session_manager.update_session_state(session_id, "profile_collection")
            
        except Exception as e:
            logger.error(f"Error processing combined response: {e}")
    
    def _is_profile_complete(self, profile_data: Dict[str, Any]) -> bool:
        """
        Check if profile collection is complete.
        
        Args:
            profile_data: Profile data dictionary
            
        Returns:
            True if profile is complete, False otherwise
        """
        required_fields = ["name", "age", "primary_cuisine", "medical_conditions", "height", "weight"]
        # Check if all required fields are present (medical_conditions can be empty list)
        for field in required_fields:
            if field == "medical_conditions":
                # medical_conditions is valid if it's a list (even if empty)
                if not isinstance(profile_data.get(field), list):
                    return False
            else:
                # Other fields must have a non-null value
                if not profile_data.get(field):
                    return False
        return True
    

    def _update_profile_from_llm_response(self, session_id: str, llm_response: str) -> None:
        """
        Update profile collection from LLM extraction response.
        
        Args:
            session_id: Session identifier
            llm_response: LLM response containing extracted data
        """
        try:
            import json
            import re
            
            # Try to find and parse JSON response
            if "{" in llm_response and "}" in llm_response:
                # Extract JSON from response using regex to handle nested braces
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_matches = re.findall(json_pattern, llm_response)
                
                if json_matches:
                    json_str = json_matches[0]  # Take the first JSON object found
                    try:
                        extracted_data = json.loads(json_str)
                        
                        # Update session with extracted data
                        for field, value in extracted_data.items():
                            if value is not None and value != "null" and value != "":
                                if field == "medical_conditions" and isinstance(value, list):
                                    # Handle medical conditions
                                    current_conditions = self.session_manager.get_profile_collection(session_id).get("medical_conditions", [])
                                    current_conditions.extend(value)
                                    self.session_manager.update_profile_collection(session_id, field, current_conditions)
                                else:
                                    # Handle other fields
                                    self.session_manager.update_profile_collection(session_id, field, value)
                                    
                    except json.JSONDecodeError as json_error:
                        logger.warning(f"Failed to parse JSON from LLM response: {json_error}")
                        logger.debug(f"JSON string was: {json_str}")
            else:
                logger.debug("No JSON structure found in LLM response")
            
        except Exception as e:
            logger.error(f"Error updating profile from LLM response: {e}")
            logger.debug(f"LLM response was: {llm_response[:200]}...")
            
    def _create_user_profile_from_session(self, profile_data: Dict[str, Any]) -> Optional[UserProfile]:
        """
        Create UserProfile object from session data.
        
        Args:
            profile_data: Profile data from session
            
        Returns:
            UserProfile object or None if invalid
        """
        try:
            # Validate required fields
            if not profile_data:
                logger.error("Profile data is None or empty")
                return None
                
            required_fields = ["name", "age", "primary_cuisine"]
            for field in required_fields:
                if not profile_data.get(field):
                    logger.error(f"Missing required field: {field}")
                    return None
            
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
            logger.error(f"Error creating user profile: {e}")
            logger.error(f"Profile data: {profile_data}")
            return None
    
    def _extract_name_from_message(self, message: str) -> Optional[str]:
        """
        Extract name from user message using keyword-based fallback.
        Note: This method is now deprecated in favor of the combined approach.
        """
        # This method is kept for compatibility but is no longer used
        # Name extraction is now handled in the combined prompt
        return None
    
    @tool
    def check_user_exists_tool(self, name: str) -> bool:
        """Check if a user exists in the database by name."""
        return self.neo4j_service.check_user_exists(name)
    
    def get_existing_user_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get existing user profile from Neo4j database.
        
        Args:
            name: User's name
            
        Returns:
            Profile data if exists, None otherwise
        """
        try:
            user_profile = self.neo4j_service.get_user_by_name(name)
            if user_profile:
                # Convert UserProfile to dictionary
                profile_data = {
                    "name": user_profile.name,
                    "age": user_profile.age,
                    "height": user_profile.height,
                    "weight": user_profile.weight,
                    "medical_conditions": user_profile.medical_conditions or [],
                    "primary_cuisine": user_profile.primary_cuisine,
                    "secondary_cuisine": user_profile.secondary_cuisine,
                    "user_id": user_profile.user_id,
                    "is_complete": True
                }
                logger.info(f"Found existing profile for {name}: {profile_data}")
                return profile_data
            else:
                logger.info(f"No existing profile found for {name}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting existing user profile: {e}")
            return None

    @tool
    def create_user_profile_tool(self, name: str, age: int, primary_cuisine: str, 
                                secondary_cuisine: str = None, height: float = None, 
                                weight: float = None, medical_conditions: List[dict] = None) -> str:
        """Create a new user profile in the database."""
        try:
            medical_conditions_list = []
            if medical_conditions:
                for condition_data in medical_conditions:
                    medical_conditions_list.append(MedicalCondition(
                        condition=condition_data["condition"],
                        intensity=condition_data["intensity"]
                    ))
            
            profile = UserProfile(
                name=name, age=age, height=height, weight=weight,
                medical_conditions=medical_conditions_list,
                primary_cuisine=primary_cuisine, secondary_cuisine=secondary_cuisine
            )
            
            user_id = self.neo4j_service.create_user_profile(profile)
            return f"Successfully created profile for {name} with ID: {user_id}"
        except Exception as e:
            return f"Error creating profile: {str(e)}"

