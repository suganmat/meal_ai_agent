"""
Data models for the meal prediction AI agent.
"""
from .user_models import UserProfile, MedicalCondition
from .meal_models import MealSuggestion, NutritionInfo, MealType, MealFeedback

__all__ = [
    "UserProfile",
    "MedicalCondition", 
    "MealSuggestion",
    "NutritionInfo",
    "MealType",
    "MealFeedback"
]