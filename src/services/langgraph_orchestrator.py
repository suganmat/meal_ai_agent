"""
LangGraph orchestrator for managing the meal prediction AI agent workflow.
"""
import logging
from typing import Dict, Any, Optional, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from ..agents.profile_collector import ProfileCollectorAgent
from ..agents.meal_suggester import MealSuggesterAgent
from ..agents.satisfaction_checker import SatisfactionCheckerAgent
from ..agents.intent_detection_agent import IntentDetectionAgent
from ..agents.normal_chat_agent import NormalChatAgent
from ..services.perplexity_client import PerplexityClient
from ..services.neo4j_service import Neo4jService
from ..utils.session_manager import SessionManager
logger = logging.getLogger(__name__)


class MealAgentOrchestrator:
    """Orchestrator for managing the meal prediction AI agent workflow."""
    
    def __init__(self):
        """Initialize the orchestrator with all required services."""
        self.perplexity_client = PerplexityClient()
        self.neo4j_service = Neo4jService()
        self.session_manager = SessionManager()
        
        # Initialize agents with shared session manager
        self.intent_detection = IntentDetectionAgent(self.perplexity_client, self.session_manager)
        self.normal_chat = NormalChatAgent(self.perplexity_client, self.session_manager)
        self.profile_collector = ProfileCollectorAgent(self.perplexity_client, self.neo4j_service, self.session_manager)
        self.meal_suggester = MealSuggesterAgent(self.perplexity_client, self.neo4j_service, self.session_manager)
        self.satisfaction_checker = SatisfactionCheckerAgent(self.perplexity_client, self.session_manager)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with 10-node architecture.
        
        Returns:
            Configured StateGraph
        """
        # Define the state schema
        class AgentState(TypedDict):
            messages: Annotated[List[Any], add_messages]
            session_id: str
            current_state: str
            user_profile: Optional[Dict[str, Any]]
            meal_data: Optional[Dict[str, Any]]
            satisfaction_data: Optional[Dict[str, Any]]
            user_name: Optional[str]
            intent: Optional[str]
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add all 8 nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("intent_detection", self._intent_detection_node)
        workflow.add_node("normal_chat", self._normal_chat_node)
        workflow.add_node("profile_collection", self._profile_collection_node)
        workflow.add_node("meal_suggestion", self._meal_suggestion_node)
        workflow.add_node("satisfaction_check", self._satisfaction_check_node)
        
        # Add conditional edges for routing
        workflow.add_conditional_edges(
            "start",
            self._route_from_start,
            {
                "intent_detection": "intent_detection",
                "profile_collection": "profile_collection",
                "meal_suggestion": "meal_suggestion",
                "satisfaction_check": "satisfaction_check",
                "normal_chat": "normal_chat"
            }
        )
        
        workflow.add_conditional_edges(
            "intent_detection",
            self._route_from_intent,
            {
                "normal_chat": "normal_chat",
                "profile_collection": "profile_collection"
            }
        )
        
        workflow.add_conditional_edges(
            "profile_collection",
            self._route_from_profile_collection,
            {
                "meal_suggestion": "meal_suggestion",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "meal_suggestion",
            self._route_from_meal_suggestion,
            {
                "satisfaction_check": "satisfaction_check"
            }
        )
        
        workflow.add_conditional_edges(
            "satisfaction_check",
            self._route_from_satisfaction,
            {
                "end": END,
                "meal_suggestion": "meal_suggestion",
                "normal_chat": "normal_chat"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("start")
        
        return workflow.compile()
    
    def _start_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start node - entry point for all conversations.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        try:
            # Just pass through to intent detection
            state["current_state"] = "start"
            return state
            
        except Exception as e:
            logger.error(f"Error in start node: {e}")
            state["current_state"] = "start"
            return state
    
    def _intent_detection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intent detection node - analyzes user input to determine intent.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        return self.intent_detection.process_node(state)
    
    def _normal_chat_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normal chat node - handles regular conversations.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        return self.normal_chat.process_node(state)
    
    
    
    
    
    def _profile_collection_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Profile collection node.
        
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
                user_message = "Hello, I'd like to get meal recommendations."
            
            # Process with profile collector (handles name extraction internally)
            response = self.profile_collector.process_message(user_message, session_id)
            
            # Update state
            state["messages"] = add_messages(state["messages"], [AIMessage(content=response)])
            
            # Update state based on profile completion status
            session = self.session_manager.get_session(session_id)
            if session:
                profile_data = self.session_manager.get_profile_collection(session_id)
                is_complete = self._is_profile_complete(profile_data)
                has_user_id = session.get("user_id") is not None
                
                if is_complete or has_user_id:
                    state["current_state"] = "meal_suggestion"
                else:
                    state["current_state"] = "profile_collection"
            else:
                state["current_state"] = "profile_collection"
            
            return state
            
        except Exception as e:
            logger.error(f"Error in profile collection node: {e}")
            state["messages"] = add_messages(state["messages"], [
                AIMessage(content="I'm sorry, I encountered an error. Please try again.")
            ])
            return state
    
    
    def _meal_suggestion_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Meal suggestion node.
        
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
                user_message = "I'd like a meal suggestion."
            
            # Process with meal suggester
            response = self.meal_suggester.process_message(user_message, session_id)
            
            # Update state
            state["messages"] = add_messages(state["messages"], [AIMessage(content=response)])
            
            # Update meal suggestion in state
            session = self.session_manager.get_session(session_id)
            if session:
                state["meal_data"] = session.get("meal_suggestion", {})
                state["current_state"] = session.get("current_state", "meal_suggestion")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in meal suggestion node: {e}")
            state["messages"] = add_messages(state["messages"], [
                AIMessage(content="I'm sorry, I encountered an error while suggesting meals. Please try again.")
            ])
            return state
    
    def _satisfaction_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Satisfaction check node.
        
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
                user_message = "How does this meal suggestion sound?"
            
            # Process with satisfaction checker
            response = self.satisfaction_checker.process_message(user_message, session_id)
            
            # Update state
            state["messages"] = add_messages(state["messages"], [AIMessage(content=response)])
            
            # Update satisfaction in state
            session = self.session_manager.get_session(session_id)
            if session:
                state["satisfaction_data"] = session.get("satisfaction", {})
                state["current_state"] = session.get("current_state", "satisfaction_check")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in satisfaction check node: {e}")
            state["messages"] = add_messages(state["messages"], [
                AIMessage(content="I'm sorry, I encountered an error. Please try again.")
            ])
            return state
    
    
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
    
    def _route_from_start(self, state: Dict[str, Any]) -> str:
        """Route from start node based on current session state."""
        session_id = state["session_id"]
        session = self.session_manager.get_session(session_id)
        
        if session:
            current_state = session.get("current_state", "initial")
            logger.info(f"Current session state: {current_state}")
            
            # If we're in the middle of profile collection, continue there
            if current_state == "profile_collection":
                logger.info("Continuing profile collection")
                return "profile_collection"
            # If we're in meal suggestion, continue there
            elif current_state == "meal_suggestion":
                logger.info("Continuing meal suggestion")
                return "meal_suggestion"
            # If we're in satisfaction check, continue there
            elif current_state == "satisfaction_check":
                logger.info("Continuing satisfaction check")
                return "satisfaction_check"
            # If we're in normal chat, continue there
            elif current_state == "normal_chat":
                logger.info("Continuing normal chat")
                return "normal_chat"
        
        # Default to intent detection for new sessions
        logger.info("Starting with intent detection")
        return "intent_detection"
    
    def _route_from_intent(self, state: Dict[str, Any]) -> str:
        """Route from intent detection based on detected intent."""
        intent = state.get("intent", "normal_chat")
        session_id = state["session_id"]
        session = self.session_manager.get_session(session_id)
        
        logger.info(f"Routing from intent detection: intent={intent}")
        
        if intent == "meal_request":
            # Check if we're in the middle of profile collection
            if session:
                profile_data = self.session_manager.get_profile_collection(session_id)
                required_fields = ["name", "age", "primary_cuisine", "medical_conditions", "height", "weight"]
                is_profile_incomplete = not all(profile_data.get(field) for field in required_fields)
                has_user_id = session.get("user_id") is not None
                
                # If profile is incomplete and we don't have a user_id, continue profile collection
                if is_profile_incomplete and not has_user_id:
                    logger.info("Profile collection in progress, routing to profile_collection")
                    return "profile_collection"
            
            logger.info("Routing to profile_collection for meal request")
            return "profile_collection"
        else:
            logger.info("Routing to normal_chat")
            return "normal_chat"
    
    
    
    def _route_from_profile_collection(self, state: Dict[str, Any]) -> str:
        """Route from profile collection based on completion status."""
        session_id = state["session_id"]
        session = self.session_manager.get_session(session_id)
        
        if not session:
            return "end"
        
        # Check if profile collection is complete
        profile_data = self.session_manager.get_profile_collection(session_id)
        is_complete = self._is_profile_complete(profile_data)
        
        # Also check if we have a user_id (indicates profile was created)
        has_user_id = session.get("user_id") is not None
        
        if is_complete or has_user_id:
            logger.info("Profile collection complete, routing to meal suggestion")
            return "meal_suggestion"
        else:
            # Profile incomplete - end workflow to wait for next user message
            logger.info("Profile collection incomplete, ending workflow to wait for next user message")
            return "end"
    
    
    def _route_from_meal_suggestion(self, state: Dict[str, Any]) -> str:
        """Route from meal suggestion based on completion."""
        session_id = state["session_id"]
        session = self.session_manager.get_session(session_id)
        
        if not session:
            return "satisfaction_check"  # End workflow if no session
        
        # Check if meal suggestion is complete
        if session.get("current_state") == "satisfaction_check":
            return "satisfaction_check"
        else:
            return "satisfaction_check"  # End workflow after meal suggestion
    
    def _route_from_satisfaction(self, state: Dict[str, Any]) -> str:
        """Route from satisfaction check based on user response."""
        session_id = state["session_id"]
        session = self.session_manager.get_session(session_id)
        
        if not session:
            return "end"
        
        # Check if user is satisfied
        satisfaction = session.get("satisfaction", {})
        satisfaction_level = satisfaction.get("level")
        
        if satisfaction_level == "satisfied":
            return "normal_chat"  # ✅ Go back to normal chat instead of ending
        elif satisfaction_level == "not_satisfied":
            # Check if user wants a new suggestion
            wants_new_suggestion = satisfaction.get("wants_new_suggestion", False)
            if wants_new_suggestion:
                return "meal_suggestion"  # ✅ Provide new meal suggestion
            else:
                return "normal_chat"  # ✅ Go to normal chat if they don't want new suggestion
        
        # Check if user wants to try again (fallback for keyword matching)
        messages = state["messages"]
        if messages:
            latest_message = messages[-1]
            if isinstance(latest_message, HumanMessage):
                message_content = latest_message.content.lower()
                if any(keyword in message_content for keyword in ["try again", "different", "another", "new meal"]):
                    return "meal_suggestion"
                if any(keyword in message_content for keyword in ["hello", "hi", "how are you", "what can you do"]):
                    return "normal_chat"
        
        return "end"
    
    def process_message(self, message: str, session_id: str) -> str:
        """
        Process a user message through the workflow.
        
        Args:
            message: User's message
            session_id: Session identifier
            
        Returns:
            Agent's response
        """
        try:
            # Ensure session exists
            session = self.session_manager.get_session(session_id)
            if not session:
                # Create new session and use the returned session_id
                session_id = self.session_manager.create_session()
                session = self.session_manager.get_session(session_id)
            
            # Add user message to session history
            self.session_manager.add_message_to_history(session_id, "user", message)
            
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "session_id": session_id,
                "current_state": "start",
                "user_profile": None,
                "meal_data": None,
                "satisfaction_data": None,
                "user_name": None,
                "intent": None
            }
            
            # Run the workflow with increased recursion limit and debugging
            try:
                result = self.workflow.invoke(initial_state, config={"recursion_limit": 50})
                logger.info(f"Workflow completed successfully. Final state: {result.get('current_state', 'unknown')}")
                logger.info(f"Workflow result keys: {list(result.keys())}")
                logger.info(f"Workflow result: {result}")
                
                # Ensure we have a valid result
                if not result:
                    logger.error("Workflow returned empty result")
                    return "I'm sorry, I encountered an error. Please try again."
                
                # Update session state with workflow result
                session = self.session_manager.get_session(session_id)
                if session and result:
                    session["current_state"] = result.get("current_state", "initial")
                    session["intent"] = result.get("intent")
                    session["user_name"] = result.get("user_name")
                    session["user_profile"] = result.get("user_profile")
                    session["profile_collection"] = result.get("profile_collection", {})
                    session["meal_data"] = result.get("meal_data")
                    session["satisfaction_data"] = result.get("satisfaction_data")
                    logger.info(f"Updated session state: {session}")
                    
            except Exception as workflow_error:
                logger.error(f"Workflow execution error: {workflow_error}")
                # Return a fallback response
                return "I'm experiencing some technical difficulties. Please try again later."
            
            # Get the latest AI message
            messages = result["messages"]
            if messages:
                latest_message = messages[-1]
                if isinstance(latest_message, AIMessage):
                    return latest_message.content
            
            return "I'm sorry, I couldn't process your request. Please try again."
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, I encountered an error. Please try again."
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get current session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session state data
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return {}
        
        return {
            "session_id": session_id,
            "current_state": session.get("current_state", "initial"),
            "user_id": session.get("user_id"),
            "profile_collection": session.get("profile_collection", {}),
            "meal_data": session.get("meal_suggestion", {}),
            "satisfaction": session.get("satisfaction", {}),
            "conversation_history": session.get("conversation_history", [])
        }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if cleared successfully
        """
        return self.session_manager.clear_session(session_id)
