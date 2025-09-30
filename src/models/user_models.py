"""
User profile models for the meal prediction AI agent.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class MedicalCondition(BaseModel):
    """Represents a medical condition with intensity level."""
    condition: str = Field(..., description="Name of the medical condition")
    intensity: str = Field(..., description="Intensity level: mild, moderate, or severe")


class UserProfile(BaseModel):
    """Complete user profile for meal recommendations."""
    user_id: Optional[str] = Field(None, description="Unique user identifier")
    name: str = Field(..., description="User's name")
    age: int = Field(..., ge=13, le=120, description="User's age (13-120)")
    height: Optional[float] = Field(None, ge=50, le=300, description="Height in cm")
    weight: Optional[float] = Field(None, ge=20, le=500, description="Weight in kg")
    medical_conditions: List[MedicalCondition] = Field(default_factory=list, description="List of medical conditions")
    primary_cuisine: str = Field(..., description="Primary cuisine preference")
    secondary_cuisine: Optional[str] = Field(None, description="Secondary cuisine preference")
    
    @property
    def bmi(self) -> Optional[float]:
        """Calculate BMI if height and weight are available."""
        if self.height and self.weight:
            return round(self.weight / ((self.height / 100) ** 2), 2)
        return None
    
    @property
    def bmi_category(self) -> Optional[str]:
        """Get BMI category based on calculated BMI."""
        if self.bmi is None:
            return None
        
        if self.bmi < 18.5:
            return "underweight"
        elif self.bmi < 25:
            return "normal"
        elif self.bmi < 30:
            return "overweight"
        else:
            return "obese"
    
    def has_medical_condition(self, condition: str) -> bool:
        """Check if user has a specific medical condition."""
        return any(mc.condition.lower() == condition.lower() for mc in self.medical_conditions)
    
    def get_condition_intensity(self, condition: str) -> Optional[str]:
        """Get intensity level for a specific medical condition."""
        for mc in self.medical_conditions:
            if mc.condition.lower() == condition.lower():
                return mc.intensity
        return None
