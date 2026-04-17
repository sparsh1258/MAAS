from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    height_cm = Column(Float, nullable=False)
    weight_kg = Column(Float, nullable=False)
    region = Column(String, nullable=False)
    weeks_pregnant = Column(Integer, nullable=False)

    history_diabetes = Column(Boolean, default=False)
    history_hypertension = Column(Boolean, default=False)
    history_preeclampsia = Column(Boolean, default=False)
    history_prev_comp = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    @property
    def trimester(self):
        if self.weeks_pregnant <= 12:
            return 1
        elif self.weeks_pregnant <= 26:
            return 2
        else:
            return 3
    @property
    def checkin_frequency(self):
        if self.trimester == 1:
            return "har 3 din mein"
        elif self.trimester == 2:
            return "roz"
        else:
            return "din mein 2 baar"
    daily_checkins = relationship("DailyCheckin", back_populates="user")
    checkins_3day = relationship("Checkin3Day", back_populates="user")

class DailyCheckin(Base):
    __tablename__ = "daily_checkins"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    weeks_pregnant_at_checkin = Column(Integer, nullable=False)
    bp_systolic = Column(Integer, nullable=False)
    bp_diastolic = Column(Integer, nullable=False)

    kick_count = Column(Integer, nullable=True)
    kick_count_normal = Column(Boolean, default=True)

    symptom_headache = Column(Boolean, default=False)
    symptom_blurred_vision = Column(Boolean, default=False)
    symptom_swelling = Column(Boolean, default=False)
    symptom_abdominal_pain = Column(Boolean, default=False)
    symptom_bleeding = Column(Boolean, default=False)
    symptom_dizziness = Column(Boolean, default=False)

    meals_count = Column(Integer, nullable=False)
    water_litres = Column(Float, nullable=False)
    sleep_hours = Column(Float, nullable=False)

    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("UserProfile", back_populates="daily_checkins")

class Checkin3Day(Base):
    __tablename__ = "checkins_3day"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False)
    weeks_pregnant_at_checkin = Column(Integer, nullable=False)
    weight_kg = Column(Float, nullable=False)
    energy_level = Column(Integer, nullable=False)
    breathlessness = Column(Integer, nullable=False)

    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("UserProfile", back_populates="checkins_3day")


class AuthAccount(Base):
    __tablename__ = "auth_accounts"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, index=True)
    display_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    patient_reviews = relationship("PatientReview", back_populates="reviewed_by")
    coordinator_tasks = relationship("CoordinatorTask", back_populates="owner")


class PatientReview(Base):
    __tablename__ = "patient_reviews"
    __table_args__ = (
        UniqueConstraint("patient_id", name="uq_patient_reviews_patient"),
    )

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False, index=True)
    reviewed_by_id = Column(Integer, ForeignKey("auth_accounts.id"), nullable=True)
    notes = Column(Text, nullable=True)
    reviewed = Column(Boolean, default=False)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    urgency_override = Column(String, nullable=True)
    escalated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    patient = relationship("UserProfile")
    reviewed_by = relationship("AuthAccount", back_populates="patient_reviews")


class CoordinatorTask(Base):
    __tablename__ = "coordinator_tasks"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("user_profiles.id"), nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("auth_accounts.id"), nullable=True)
    task_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    priority = Column(String, nullable=False, default="routine")
    status = Column(String, nullable=False, default="open")
    due_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    patient = relationship("UserProfile")
    owner = relationship("AuthAccount", back_populates="coordinator_tasks")
