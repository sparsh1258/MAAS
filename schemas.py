from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import Optional, List
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

    @field_validator('kick_count')
    @classmethod
    def kick_required_after_first_trimester(cls, value: Optional[int], info: ValidationInfo):
        weeks = info.data.get('weeks_pregnant_at_checkin', 0)
        if weeks > 12 and value is None:
            raise ValueError('2nd aur 3rd trimester mein kick count mandatory hai')
        return value

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

class DiagnosisResponse(BaseModel):
    user_id: int
    predicted_condition: str
    urgency: str
    rationale: str
    reward: float
    risk_flags: List[str]
    history_flags: List[str]
    diet_advice: List[str]
    days_of_data: int
    latest_checkin_at: Optional[datetime] = None
