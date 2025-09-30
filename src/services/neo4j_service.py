"""
Neo4j service for user profile management.
"""
import os
from typing import Optional, Dict, Any, List
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, TransientError
import logging
from ..models.user_models import UserProfile, MedicalCondition

logger = logging.getLogger(__name__)


class Neo4jService:
    """Service for managing user profiles in Neo4j database."""
    
    def __init__(self):
        """Initialize Neo4j connection."""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j database")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Could not connect to Neo4j database: {e}")
    
    def close(self) -> None:
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def check_user_exists(self, name: str) -> bool:
        """
        Check if a user exists in the database by name.
        
        Args:
            name: User's name to check
            
        Returns:
            True if user exists, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (u:User {name: $name}) RETURN u.name as name",
                    name=name.strip().lower()
                )
                return result.single() is not None
        except Exception as e:
            logger.error(f"Error checking if user exists: {e}")
            return False
    
    def create_user_profile(self, profile: UserProfile) -> str:
        """
        Create a new user profile in the database.
        
        Args:
            profile: UserProfile object to create
            
        Returns:
            User ID of the created profile
        """
        try:
            with self.driver.session() as session:
                # Create user node
                result = session.run(
                    """
                    CREATE (u:User {
                        user_id: $user_id,
                        name: $name,
                        age: $age,
                        height: $height,
                        weight: $weight,
                        primary_cuisine: $primary_cuisine,
                        secondary_cuisine: $secondary_cuisine,
                        bmi: $bmi,
                        created_at: datetime()
                    })
                    RETURN u.user_id as user_id
                    """,
                    user_id=profile.user_id or f"user_{profile.name.lower().replace(' ', '_')}",
                    name=profile.name.strip().lower(),
                    age=profile.age,
                    height=profile.height,
                    weight=profile.weight,
                    primary_cuisine=profile.primary_cuisine,
                    secondary_cuisine=profile.secondary_cuisine,
                    bmi=profile.bmi
                )
                
                user_id = result.single()["user_id"]
                
                # Create medical condition nodes and relationships
                for condition in profile.medical_conditions:
                    session.run(
                        """
                        MATCH (u:User {user_id: $user_id})
                        CREATE (u)-[:HAS_CONDITION]->(c:MedicalCondition {
                            condition: $condition,
                            intensity: $intensity,
                            created_at: datetime()
                        })
                        """,
                        user_id=user_id,
                        condition=condition.condition,
                        intensity=condition.intensity
                    )
                
                logger.info(f"Created user profile for {profile.name} with ID: {user_id}")
                return user_id
                
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            raise RuntimeError(f"Failed to create user profile: {e}")
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile by user ID.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            UserProfile object if found, None otherwise
        """
        try:
            with self.driver.session() as session:
                # Get user data
                user_result = session.run(
                    """
                    MATCH (u:User {user_id: $user_id})
                    RETURN u.user_id as user_id, u.name as name, u.age as age,
                           u.height as height, u.weight as weight,
                           u.primary_cuisine as primary_cuisine,
                           u.secondary_cuisine as secondary_cuisine
                    """,
                    user_id=user_id
                )
                
                user_data = user_result.single()
                if not user_data:
                    return None
                
                # Get medical conditions
                conditions_result = session.run(
                    """
                    MATCH (u:User {user_id: $user_id})-[:HAS_CONDITION]->(c:MedicalCondition)
                    RETURN c.condition as condition, c.intensity as intensity
                    """,
                    user_id=user_id
                )
                
                medical_conditions = [
                    MedicalCondition(condition=row["condition"], intensity=row["intensity"])
                    for row in conditions_result
                ]
                
                return UserProfile(
                    user_id=user_data["user_id"],
                    name=user_data["name"],
                    age=user_data["age"],
                    height=user_data["height"],
                    weight=user_data["weight"],
                    medical_conditions=medical_conditions,
                    primary_cuisine=user_data["primary_cuisine"],
                    secondary_cuisine=user_data["secondary_cuisine"]
                )
                
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return None
    
    def get_user_by_name(self, name: str) -> Optional[UserProfile]:
        """
        Retrieve a user profile by name.
        
        Args:
            name: User's name to retrieve
            
        Returns:
            UserProfile object if found, None otherwise
        """
        try:
            with self.driver.session() as session:
                # Get user data by name
                user_result = session.run(
                    """
                    MATCH (u:User {name: $name})
                    RETURN u.user_id as user_id, u.name as name, u.age as age,
                           u.height as height, u.weight as weight,
                           u.primary_cuisine as primary_cuisine,
                           u.secondary_cuisine as secondary_cuisine
                    """,
                    name=name.strip().lower()
                )
                
                user_data = user_result.single()
                if not user_data:
                    return None
                
                # Get medical conditions
                conditions_result = session.run(
                    """
                    MATCH (u:User {user_id: $user_id})-[:HAS_CONDITION]->(c:MedicalCondition)
                    RETURN c.condition as condition, c.intensity as intensity
                    """,
                    user_id=user_data["user_id"]
                )
                
                medical_conditions = [
                    MedicalCondition(condition=row["condition"], intensity=row["intensity"])
                    for row in conditions_result
                ]
                
                return UserProfile(
                    user_id=user_data["user_id"],
                    name=user_data["name"],
                    age=user_data["age"],
                    height=user_data["height"],
                    weight=user_data["weight"],
                    medical_conditions=medical_conditions,
                    primary_cuisine=user_data["primary_cuisine"],
                    secondary_cuisine=user_data["secondary_cuisine"]
                )
                
        except Exception as e:
            logger.error(f"Error retrieving user by name: {e}")
            return None
    
    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a user profile with new information.
        
        Args:
            user_id: User ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Build dynamic update query
                set_clauses = []
                params = {"user_id": user_id}
                
                for key, value in updates.items():
                    if key in ["name", "age", "height", "weight", "primary_cuisine", "secondary_cuisine"]:
                        set_clauses.append(f"u.{key} = ${key}")
                        params[key] = value
                
                if not set_clauses:
                    return False
                
                # Add updated timestamp
                set_clauses.append("u.updated_at = datetime()")
                
                query = f"""
                MATCH (u:User {{user_id: $user_id}})
                SET {', '.join(set_clauses)}
                RETURN u.user_id as user_id
                """
                
                result = session.run(query, **params)
                updated = result.single() is not None
                
                if updated:
                    logger.info(f"Updated user profile {user_id}")
                
                return updated
                
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False
    
    def delete_user_profile(self, user_id: str) -> bool:
        """
        Delete a user profile and all related data.
        
        Args:
            user_id: User ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Delete user and all relationships
                result = session.run(
                    """
                    MATCH (u:User {user_id: $user_id})
                    DETACH DELETE u
                    RETURN count(u) as deleted_count
                    """,
                    user_id=user_id
                )
                
                deleted_count = result.single()["deleted_count"]
                success = deleted_count > 0
                
                if success:
                    logger.info(f"Deleted user profile {user_id}")
                
                return success
                
        except Exception as e:
            logger.error(f"Error deleting user profile: {e}")
            return False
    
    def get_all_users(self) -> List[UserProfile]:
        """
        Get all user profiles from the database.
        
        Returns:
            List of UserProfile objects
        """
        try:
            with self.driver.session() as session:
                # Get all users
                users_result = session.run(
                    """
                    MATCH (u:User)
                    RETURN u.user_id as user_id, u.name as name, u.age as age,
                           u.height as height, u.weight as weight,
                           u.primary_cuisine as primary_cuisine,
                           u.secondary_cuisine as secondary_cuisine
                    """
                )
                
                users = []
                for user_data in users_result:
                    # Get medical conditions for each user
                    conditions_result = session.run(
                        """
                        MATCH (u:User {user_id: $user_id})-[:HAS_CONDITION]->(c:MedicalCondition)
                        RETURN c.condition as condition, c.intensity as intensity
                        """,
                        user_id=user_data["user_id"]
                    )
                    
                    medical_conditions = [
                        MedicalCondition(condition=row["condition"], intensity=row["intensity"])
                        for row in conditions_result
                    ]
                    
                    users.append(UserProfile(
                        user_id=user_data["user_id"],
                        name=user_data["name"],
                        age=user_data["age"],
                        height=user_data["height"],
                        weight=user_data["weight"],
                        medical_conditions=medical_conditions,
                        primary_cuisine=user_data["primary_cuisine"],
                        secondary_cuisine=user_data["secondary_cuisine"]
                    ))
                
                return users
                
        except Exception as e:
            logger.error(f"Error retrieving all users: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
