"""
Meal suggestion models for the meal prediction AI agent.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class MealType(str, Enum):
    """Types of meals."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class NutritionInfo(BaseModel):
    """Nutritional information for a meal."""
    calories: int = Field(..., ge=0, description="Calories per serving")
    protein_g: float = Field(..., ge=0, description="Protein in grams")
    carbs_g: float = Field(..., ge=0, description="Carbohydrates in grams")
    fat_g: float = Field(..., ge=0, description="Fat in grams")
    fiber_g: float = Field(..., ge=0, description="Fiber in grams")
    sodium_mg: Optional[float] = Field(None, ge=0, description="Sodium in milligrams")
    sugar_g: Optional[float] = Field(None, ge=0, description="Sugar in grams")
    
    @property
    def total_macros(self) -> float:
        """Calculate total macronutrients (protein + carbs + fat)."""
        return self.protein_g + self.carbs_g + self.fat_g


class MealSuggestion(BaseModel):
    """Complete meal suggestion with all details."""
    name: str = Field(..., description="Name of the meal")
    meal_type: MealType = Field(..., description="Type of meal")
    preparation_time: int = Field(..., ge=1, le=300, description="Preparation time in minutes")
    cooking_time: Optional[int] = Field(None, ge=0, le=300, description="Cooking time in minutes")
    servings: int = Field(default=1, ge=1, le=20, description="Number of servings")
    difficulty: str = Field(default="easy", description="Difficulty level: easy, medium, hard")
    nutrition: NutritionInfo = Field(..., description="Nutritional information")
    ingredients: List[str] = Field(..., description="List of ingredients with quantities")
    instructions: List[str] = Field(..., description="Step-by-step cooking instructions")
    cuisine_type: str = Field(..., description="Type of cuisine")
    dietary_tags: List[str] = Field(default_factory=list, description="Dietary tags (e.g., gluten-free, vegan)")
    health_benefits: List[str] = Field(default_factory=list, description="Health benefits of this meal")
    
    @property
    def total_time(self) -> int:
        """Total time including preparation and cooking."""
        return self.preparation_time + (self.cooking_time or 0)
    
    def get_nutrition_per_serving(self) -> NutritionInfo:
        """Get nutrition information per single serving."""
        if self.servings == 1:
            return self.nutrition
        
        return NutritionInfo(
            calories=int(self.nutrition.calories / self.servings),
            protein_g=round(self.nutrition.protein_g / self.servings, 2),
            carbs_g=round(self.nutrition.carbs_g / self.servings, 2),
            fat_g=round(self.nutrition.fat_g / self.servings, 2),
            fiber_g=round(self.nutrition.fiber_g / self.servings, 2),
            sodium_mg=round(self.nutrition.sodium_mg / self.servings, 2) if self.nutrition.sodium_mg else None,
            sugar_g=round(self.nutrition.sugar_g / self.servings, 2) if self.nutrition.sugar_g else None
        )


class MealFeedback(BaseModel):
    """User feedback on a meal suggestion."""
    meal_id: str = Field(..., description="ID of the meal suggestion")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback_text: Optional[str] = Field(None, description="Additional feedback text")
    would_make_again: bool = Field(..., description="Whether user would make this meal again")
    modifications_made: List[str] = Field(default_factory=list, description="Any modifications user made")
    cooking_issues: List[str] = Field(default_factory=list, description="Any issues encountered while cooking")
