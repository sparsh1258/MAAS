from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class UserProfileCreate(BaseModel):
    name: str
    age: int = Field(..., ge=15, le=55)
    height_cm: float = Field(..., ge=100, le=220)
    weight_kg: float = Field(..., ge=30, le=150)
    region: str
    weeks_pregnant: int = Field(..., ge=1, le=42)
    history_diabetes: bool = False
    history_hypertension: bool = False
    history_preeclampsia: bool = False
    history_prev_comp: bool = False

class UserProfileResponse(UserProfileCreate):
    id: int
    trimester: int
    checkin_frequency: str
    created_at: datetime

    class Config:
        from_attributes = True  

class DailyCheckinCreate(BaseModel):
    user_id: int
    weeks_pregnant_at_checkin: int = Field(..., ge=1, le=42)
    bp_systolic: int = Field(..., ge=60, le=250)
    bp_diastolic: int = Field(..., ge=40, le=150)
    kick_count: Optional[int] = None
    kick_count_normal: Optional[bool] = None
    symptom_headache: bool = False
    symptom_blurred_vision: bool = False
    symptom_swelling: bool = False
    symptom_abdominal_pain: bool = False
    symptom_bleeding: bool = False
    symptom_dizziness: bool = False
    meals_count: int = Field(..., ge=0, le=5)
    water_litres: float = Field(..., ge=0, le=10)
    sleep_hours: float = Field(..., ge=0, le=24)
    notes: Optional[str] = None 

    @field_validator("kick_count")
    def kick_required_after_first_trimester(cls, v, values):
            weeks = values.get("weeks_pregnant_at_checkin", 0)
            if weeks > 12 and v is None:
                raise ValueError("2nd aur 3rd trimester mein kick count mandatory hai")
            return v

class DailyCheckinResponse(DailyCheckinCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Checkin3DayCreate(BaseModel):
    user_id: int
    weeks_pregnant_at_checkin: int = Field(..., ge=1, le=42)
    weight_kg: float = Field(..., ge=30, le=150)
    energy_level: int = Field(..., ge=1, le=10)
    breathlessness: int = Field(..., ge=1, le=10)
    notes: Optional[str] = None

class Checkin3DayResponse(Checkin3DayCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True