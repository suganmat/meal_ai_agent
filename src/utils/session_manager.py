"""
Session management utilities for the meal prediction AI agent.
"""
import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions for the meal prediction AI agent."""
    
    def __init__(self, session_timeout: int = 3600):
        """
        Initialize session manager.
        
        Args:
            session_timeout: Session timeout in seconds (default: 1 hour)
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = session_timeout
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self._cleanup_thread.start()
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user ID to associate with session
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        with self._lock:
            self.sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "data": {},
                "conversation_history": [],
                "current_state": "initial",
                "profile_collection": {
                    "name": None,
                    "age": None,
                    "height": None,
                    "weight": None,
                    "medical_conditions": [],
                    "primary_cuisine": None,
                    "secondary_cuisine": None,
                    "is_complete": False
                },
                "meal_suggestion": {
                    "meal_type": None,
                    "suggestion": None,
                    "feedback": None
                }
            }
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by session ID.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session data or None if not found/expired
        """
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check if session is expired
            if self._is_session_expired(session):
                del self.sessions[session_id]
                logger.info(f"Session expired and removed: {session_id}")
                return None
            
            # Update last accessed time
            session["last_accessed"] = datetime.now()
            return session.copy()
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session ID to update
            data: Data to update
            
        Returns:
            True if update successful, False otherwise
        """
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            
            # Check if session is expired
            if self._is_session_expired(session):
                del self.sessions[session_id]
                return False
            
            # Update session data
            session["data"].update(data)
            session["last_accessed"] = datetime.now()
            
            logger.debug(f"Updated session {session_id}")
            return True
    
    def update_session_state(self, session_id: str, state: str) -> bool:
        """
        Update the current state of a session.
        
        Args:
            session_id: Session ID to update
            state: New state
            
        Returns:
            True if update successful, False otherwise
        """
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            
            if self._is_session_expired(session):
                del self.sessions[session_id]
                return False
            
            session["current_state"] = state
            session["last_accessed"] = datetime.now()
            
            logger.debug(f"Updated session {session_id} state to {state}")
            return True
    
    def add_message_to_history(self, session_id: str, role: str, content: str) -> bool:
        """
        Add a message to the conversation history.
        
        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            
            if self._is_session_expired(session):
                del self.sessions[session_id]
                return False
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            session["conversation_history"].append(message)
            session["last_accessed"] = datetime.now()
            
            # Keep only last 50 messages to prevent memory issues
            if len(session["conversation_history"]) > 50:
                session["conversation_history"] = session["conversation_history"][-50:]
            
            return True
    
    def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        history = session["conversation_history"]
        if limit:
            return history[-limit:]
        return history
    
    
    def _validate_session(self, session_id: str) -> bool:
        """Validate that session exists and is not expired."""
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            if self._is_session_expired(session):
                del self.sessions[session_id]
                return False
            
            return True
    
    def update_profile_collection(self, session_id: str, field: str, value: Any) -> bool:
        """
        Update profile collection data for a session.
        
        Args:
            session_id: Session ID
            field: Field to update
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        if not self._validate_session(session_id):
            return False
        
        with self._lock:
            session = self.sessions[session_id]
            
            if field in session["profile_collection"]:
                session["profile_collection"][field] = value
                session["last_accessed"] = datetime.now()
                
                # Check if profile collection is complete
                profile = session["profile_collection"]
                is_complete = all([
                    profile["name"],
                    profile["age"],
                    profile["primary_cuisine"]
                ])
                profile["is_complete"] = is_complete
                
                return True
            
            return False
    
    def get_profile_collection(self, session_id: str) -> Dict[str, Any]:
        """
        Get profile collection data for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Profile collection data
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        return session["profile_collection"].copy()
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            True if session was cleared, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Cleared session: {session_id}")
                return True
            return False
    
    def clear_all_sessions(self) -> int:
        """
        Clear all sessions.
        
        Returns:
            Number of sessions cleared
        """
        with self._lock:
            count = len(self.sessions)
            self.sessions.clear()
            logger.info(f"Cleared all sessions: {count}")
            return count
    
    def get_session_count(self) -> int:
        """
        Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        with self._lock:
            return len(self.sessions)
    
    def _is_session_expired(self, session: Dict[str, Any]) -> bool:
        """
        Check if a session is expired.
        
        Args:
            session: Session data
            
        Returns:
            True if expired, False otherwise
        """
        last_accessed = session["last_accessed"]
        return datetime.now() - last_accessed > timedelta(seconds=self.session_timeout)
    
    def _cleanup_expired_sessions(self):
        """Background thread to clean up expired sessions."""
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                
                with self._lock:
                    expired_sessions = []
                    for session_id, session in self.sessions.items():
                        if self._is_session_expired(session):
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                        logger.info(f"Cleaned up expired session: {session_id}")
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        with self._lock:
            now = datetime.now()
            active_sessions = 0
            recent_sessions = 0
            
            for session in self.sessions.values():
                if not self._is_session_expired(session):
                    active_sessions += 1
                    if now - session["last_accessed"] < timedelta(minutes=10):
                        recent_sessions += 1
            
            return {
                "total_sessions": len(self.sessions),
                "active_sessions": active_sessions,
                "recent_sessions": recent_sessions,
                "session_timeout": self.session_timeout
            }
