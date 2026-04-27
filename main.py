# ============================================================
#  EduPlatform Backend — single file
#  Stack: FastAPI + SQLite + Gemini 1.5 Flash
#  Author: generated for Ansh
# ============================================================

import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Any

# ── FastAPI / Pydantic ────────────────────────────────────
from fastapi import (
    FastAPI, Depends, HTTPException, status, Body
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from pydantic_settings import BaseSettings

# ── Auth / Security ───────────────────────────────────────
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Database ──────────────────────────────────────────────
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    DateTime, Boolean, ForeignKey, Float, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func

# ── Gemini ────────────────────────────────────────────────
import google.generativeai as genai


# ╔══════════════════════════════════════════════════════════╗
# ║  1. CONFIG                                               ║
# ╚══════════════════════════════════════════════════════════╝

class Settings(BaseSettings):
    gemini_api_key: str = ""
    database_url: str = "sqlite:///./edu_platform.db"
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_MIN_32_CHARS!!"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()


# ╔══════════════════════════════════════════════════════════╗
# ║  2. DATABASE                                             ║
# ╚══════════════════════════════════════════════════════════╝

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── ORM Models ────────────────────────────────────────────

class UserModel(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String(100), nullable=False)
    email           = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())

    syllabi  = relationship("SyllabusModel",        back_populates="owner",  cascade="all, delete-orphan")
    sessions = relationship("LectureSessionModel",  back_populates="user",   cascade="all, delete-orphan")
    alarms   = relationship("AlarmModel",           back_populates="user",   cascade="all, delete-orphan")
    progress = relationship("ProgressModel",        back_populates="user",   uselist=False, cascade="all, delete-orphan")


class SyllabusModel(Base):
    __tablename__ = "syllabi"
    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    subject      = Column(String(200), nullable=False)
    exam_date    = Column(DateTime(timezone=True), nullable=True)
    topics       = Column(JSON, nullable=False)
    total_topics = Column(Integer, default=0)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("UserModel",      back_populates="syllabi")
    plan  = relationship("StudyPlanModel", back_populates="syllabus", uselist=False, cascade="all, delete-orphan")


class StudyPlanModel(Base):
    __tablename__ = "study_plans"
    id           = Column(Integer, primary_key=True, index=True)
    syllabus_id  = Column(Integer, ForeignKey("syllabi.id"), nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    schedule     = Column(JSON, nullable=False)
    total_days   = Column(Integer, default=0)
    summary      = Column(Text, nullable=True)

    syllabus        = relationship("SyllabusModel",    back_populates="plan")
    lectures        = relationship("LectureModel",     back_populates="plan", cascade="all, delete-orphan")
    calendar_events = relationship("CalendarEventModel", back_populates="plan", cascade="all, delete-orphan")


class LectureModel(Base):
    __tablename__ = "lectures"
    id               = Column(Integer, primary_key=True, index=True)
    plan_id          = Column(Integer, ForeignKey("study_plans.id"), nullable=False)
    topic            = Column(String(300), nullable=False)
    scheduled_date   = Column(DateTime(timezone=True), nullable=False)
    status           = Column(String(20), default="pending")   # pending | active | completed
    lecture_content  = Column(Text, nullable=True)
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    plan     = relationship("StudyPlanModel",      back_populates="lectures")
    sessions = relationship("LectureSessionModel", back_populates="lecture", cascade="all, delete-orphan")


class LectureSessionModel(Base):
    __tablename__ = "lecture_sessions"
    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    lecture_id   = Column(Integer, ForeignKey("lectures.id"), nullable=False)
    mode         = Column(String(10), default="text")   # text | voice
    started_at   = Column(DateTime(timezone=True), server_default=func.now())
    ended_at     = Column(DateTime(timezone=True), nullable=True)
    is_active    = Column(Boolean, default=True)
    messages     = Column(JSON, default=list)
    progress_pct = Column(Float, default=0.0)

    user    = relationship("UserModel",    back_populates="sessions")
    lecture = relationship("LectureModel", back_populates="sessions")
    doubts  = relationship("DoubtModel",   back_populates="session", cascade="all, delete-orphan")


class DoubtModel(Base):
    __tablename__ = "doubts"
    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("lecture_sessions.id"), nullable=False)
    question   = Column(Text, nullable=False)
    answer     = Column(Text, nullable=True)
    asked_at   = Column(DateTime(timezone=True), server_default=func.now())
    resolved   = Column(Boolean, default=False)

    session = relationship("LectureSessionModel", back_populates="doubts")


class CalendarEventModel(Base):
    __tablename__ = "calendar_events"
    id          = Column(Integer, primary_key=True, index=True)
    plan_id     = Column(Integer, ForeignKey("study_plans.id"), nullable=False)
    lecture_id  = Column(Integer, ForeignKey("lectures.id"), nullable=True)
    title       = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)
    event_date  = Column(DateTime(timezone=True), nullable=False)
    event_type  = Column(String(30), default="lecture")  # lecture | exam | revision

    plan = relationship("StudyPlanModel", back_populates="calendar_events")


class AlarmModel(Base):
    __tablename__ = "alarms"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    lecture_id     = Column(Integer, ForeignKey("lectures.id"), nullable=True)
    label          = Column(String(200), nullable=False)
    alarm_time     = Column(DateTime(timezone=True), nullable=False)
    is_active      = Column(Boolean, default=True)
    minutes_before = Column(Integer, default=10)

    user = relationship("UserModel", back_populates="alarms")


class ProgressModel(Base):
    __tablename__ = "progress"
    id                 = Column(Integer, primary_key=True, index=True)
    user_id            = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    total_lectures     = Column(Integer, default=0)
    completed_lectures = Column(Integer, default=0)
    total_doubts       = Column(Integer, default=0)
    resolved_doubts    = Column(Integer, default=0)
    streak_days        = Column(Integer, default=0)
    last_active        = Column(DateTime(timezone=True), nullable=True)
    updated_at         = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("UserModel", back_populates="progress")


# ╔══════════════════════════════════════════════════════════╗
# ║  3. AUTH UTILS                                           ║
# ╚══════════════════════════════════════════════════════════╝

pwd_context    = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserModel:
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise exc
    except JWTError:
        raise exc
    user = db.query(UserModel).filter(UserModel.id == int(user_id)).first()
    if not user:
        raise exc
    return user


# ╔══════════════════════════════════════════════════════════╗
# ║  4. GEMINI SERVICE  (structured prompt engineering)      ║
# ╚══════════════════════════════════════════════════════════╝
#
#  DESIGN PHILOSOPHY
#  -----------------
#  Instead of fine-tuning (requires GPU + data), we use:
#
#  1. PERSONA LOCK   — model is given a strict teacher identity
#                      with explicit rules it must follow.
#  2. TOPIC FENCE    — model is forbidden from going off-topic;
#                      off-topic input triggers a redirect.
#  3. LECTURE PHASES — each session has 4 structured phases
#                      (Introduction → Core Teaching → Q&A Check
#                       → Recap) so the AI always delivers a
#                      complete, coherent lesson, not random text.
#  4. OUTPUT FORMAT  — every response must follow a tagged format
#                      so the backend can parse phase / progress.
#  5. SAFETY RULES   — model cannot give unrelated advice,
#                      cannot discuss other subjects, and must
#                      stay academically accurate.
#  6. TEMPERATURE=0  — deterministic, factual, no creative drift.
# ──────────────────────────────────────────────────────────────

def _gemini_model(temperature: float = 0.0):
    """
    Always returns a model with low temperature for factual,
    consistent, on-topic lecture responses.
    temperature=0  → fully deterministic (lectures, doubts)
    temperature=0.3 → slight creativity (study plan tips only)
    """
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            max_output_tokens=1024,
        ),
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
    )


def _clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── MASTER TEACHER PERSONA ────────────────────────────────────────────────────
# This is injected at the start of EVERY lecture-related prompt.
# It is the "fine-tuning substitute" — a strict behavioral contract.

def _teacher_persona(subject: str, topic: str, mode: str = "text") -> str:
    return f"""
=== AI TEACHER IDENTITY & STRICT RULES ===

You are EduBot, an expert academic teacher specialising in {subject}.
Your ONLY job right now is to teach the topic: "{topic}".

ABSOLUTE RULES — you must follow these without exception:
1. STAY ON TOPIC: Every response must be directly about "{topic}" within {subject}.
   If the student asks about something unrelated, say:
   "That's outside today's topic. Let's stay focused on {topic} — I'll note your
   question and you can explore it after this lecture."
   Then immediately return to teaching {topic}.

2. NEVER HALLUCINATE: Only state facts you are certain about.
   If unsure, say "I want to verify this — let's reason through it together."
   Do NOT invent formulas, dates, names, or definitions.

3. STRUCTURED TEACHING — follow this lecture flow every session:
   Phase 1 — INTRODUCTION  : Define the topic, why it matters, real-world use.
   Phase 2 — CORE TEACHING : Explain concepts step by step, one concept at a time.
   Phase 3 — EXAMPLES      : Give 1-2 concrete examples per concept.
   Phase 4 — CHECK         : Ask the student a question to verify understanding.
   Phase 5 — RECAP         : Summarise key points before ending.
   Do NOT jump to Phase 5 unless all earlier phases are complete.

4. ONE CONCEPT AT A TIME: Never dump everything at once. Teach one idea,
   confirm understanding, then move to the next.

5. ACADEMIC LANGUAGE: Use proper subject-specific terminology.
   Always explain a term when you first use it.

6. NO CASUAL CHAT: Do not discuss movies, sports, personal topics,
   current events, other subjects, or anything outside {subject} / {topic}.

7. ENCOURAGE BUT DON'T FLATTER: Say "Good thinking!" only when the student
   actually answers correctly. Never say "Great question!" to every message.

8. {"VOICE MODE — keep each response under 80 words. Speak in short, clear sentences. No bullet points or markdown." if mode == "voice" else "TEXT MODE — use markdown formatting (headers, bold, bullets) to structure your response clearly."}

9. END EACH RESPONSE with one of:
   [TEACHING] — if you are in the middle of explaining a concept
   [CHECK]    — if you just asked the student a comprehension question
   [RECAP]    — if you are summarising at the end of the lecture

=== END OF RULES ===
"""


# ── STUDY PLAN GENERATOR ──────────────────────────────────────────────────────

def ai_generate_study_plan(subject: str, topics: List[str],
                            start_date: datetime,
                            exam_date: Optional[datetime],
                            daily_hours: int = 2) -> dict:
    """Generate a structured, difficulty-weighted day-by-day study schedule."""
    days_available = (exam_date - start_date).days if exam_date else len(topics) * 2

    prompt = f"""You are a professional academic curriculum planner.

TASK: Create a complete, realistic day-by-day study schedule.

INPUT:
  Subject        : {subject}
  Topics         : {json.dumps(topics)}
  Start date     : {start_date.strftime('%Y-%m-%d')}
  Exam date      : {exam_date.strftime('%Y-%m-%d') if exam_date else 'Not specified'}
  Days available : {days_available}
  Daily hours    : {daily_hours}

PLANNING RULES:
  - Assess each topic's complexity (foundational topics get 1 day, complex ones 2-3 days)
  - Topics that are prerequisites of others must come first
  - Insert a "Revision" day after every 5 lecture days
  - Leave the last 3 days before exam as pure revision / mock tests
  - Do NOT assign lectures on Sundays (rest day)
  - Each lecture day: duration_mins = daily_hours * 60 (max 180)
  - Notes must be a genuine study tip for that specific topic, not generic advice

OUTPUT: Respond ONLY with valid JSON, no extra text, no markdown fences:
{{
  "schedule": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "topic": "exact topic name from the input list",
      "duration_mins": 120,
      "notes": "specific actionable study tip for this topic",
      "type": "lecture"
    }}
  ],
  "summary": "3-sentence overview explaining the strategy behind this plan",
  "total_days": {days_available}
}}

type must be one of: "lecture" | "revision" | "exam" | "rest"
"""
    model = _gemini_model(temperature=0.3)
    response = model.generate_content(prompt)
    return json.loads(_clean_json(response.text))


# ── LECTURE INTRO GENERATOR ───────────────────────────────────────────────────

def ai_lecture_intro(subject: str, topic: str) -> str:
    """
    Generate a structured, pedagogically sound opening for a lecture.
    Follows Phase 1 (Introduction) of the teaching framework.
    """
    prompt = f"""
{_teacher_persona(subject, topic)}

TASK: Write the Phase 1 (INTRODUCTION) opening for today's lecture.

Your introduction MUST include ALL of the following — in this order:
1. A one-sentence welcome that names the topic explicitly.
2. WHAT: A precise definition of "{topic}" in 2-3 sentences using correct terminology.
3. WHY: Why this topic matters — give one real-world application or use case.
4. PREREQUISITES: Mention 1-2 concepts the student should already know
   (if this is a foundational topic, say "No prior knowledge needed.").
5. TODAY'S ROADMAP: A numbered list of exactly what will be covered in this lecture.
6. A closing line inviting the student to signal readiness:
   "Type 'ready' or ask any clarifying question to begin! 🎓"

FORMAT: Use markdown. Bold key terms on first use.
LENGTH: 200–280 words exactly.
DO NOT teach the content yet — this is only the introduction.

[TEACHING]
"""
    model = _gemini_model(temperature=0.0)
    return model.generate_content(prompt).text.strip()


# ── LECTURE CHAT ENGINE ───────────────────────────────────────────────────────

# Tracks which teaching phase a session is in based on message count
def _infer_phase(message_count: int) -> str:
    if message_count <= 2:
        return "INTRODUCTION"
    elif message_count <= 8:
        return "CORE TEACHING"
    elif message_count <= 12:
        return "EXAMPLES & PRACTICE"
    else:
        return "RECAP & WRAP-UP"


def ai_lecture_chat(subject: str, topic: str,
                    history: List[dict], user_message: str,
                    mode: str = "text") -> str:
    """
    Continue an AI lecture session with strict topic adherence and phase tracking.
    The model is instructed to follow a structured teaching progression.
    """
    message_count = len(history)
    current_phase = _infer_phase(message_count)

    # Detect off-topic attempts in student message
    off_topic_guard = f"""
IMPORTANT: The student just sent: "{user_message}"
Before responding, check: Is this message related to "{topic}" in {subject}?
- If YES → continue teaching normally as per the phase instructions below.
- If NO  → respond with exactly:
  "That's outside today's scope! Let's stay on **{topic}**. [Brief one-line redirect
   to the current teaching point]. [TEACHING]"
  Do NOT answer the off-topic question under any circumstances.
"""

    phase_instruction = {
        "INTRODUCTION": f"""
You are in Phase 1 (INTRODUCTION).
If the student says "ready" or similar, begin Phase 2 immediately.
Otherwise, answer their clarifying question about the topic briefly, then re-invite them to start.
""",
        "CORE TEACHING": f"""
You are in Phase 2 (CORE TEACHING).
- Teach ONE sub-concept of "{topic}" at a time.
- State the concept name clearly (bold it).
- Explain it in 3-5 sentences using correct terminology.
- After explaining, give ONE real-world or practical example.
- End by asking: "Does this make sense so far?" or a specific comprehension question.
- Do NOT move to the next sub-concept until the student confirms understanding.
""",
        "EXAMPLES & PRACTICE": f"""
You are in Phase 3 (EXAMPLES & PRACTICE).
- Provide a concrete worked example or problem related to "{topic}".
- Walk through the example step by step.
- Then ask the student to solve a similar small problem themselves.
- If they answer, evaluate it: if correct say why; if wrong explain the mistake clearly.
""",
        "RECAP & WRAP-UP": f"""
You are in Phase 5 (RECAP).
- Summarise all key concepts covered in this lecture as a numbered list.
- Highlight the 2-3 most important takeaways.
- Suggest what the student should review or practise next.
- End with: "Great work today! Type /end to finish the lecture. 🎓"
""",
    }.get(current_phase, "")

    system_prompt = f"""
{_teacher_persona(subject, topic, mode)}

CURRENT LECTURE PHASE: {current_phase} (message {message_count + 1} of this session)
{phase_instruction}
{off_topic_guard}
"""

    # Build Gemini history (last 12 messages for context)
    gemini_history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history[-12:]
    ]

    model = _gemini_model(temperature=0.0)

    if gemini_history:
        # Inject system context as a prepended user turn if not already set
        chat = model.start_chat(history=gemini_history)
        full_msg = f"{system_prompt}\n\nStudent message: {user_message}"
    else:
        # First real student message — full system context goes in
        chat = model.start_chat(history=[])
        full_msg = f"{system_prompt}\n\nStudent message: {user_message}"

    response = chat.send_message(full_msg)
    return response.text.strip()


# ── DOUBT SOLVER ──────────────────────────────────────────────────────────────

def ai_solve_doubt(subject: str, topic: str, question: str,
                   context: Optional[List[dict]] = None) -> str:
    """
    Answer a student's doubt with a structured, accurate explanation.
    Strictly stays within the topic boundary — will not answer unrelated questions.
    """
    ctx_str = ""
    if context:
        ctx_str = "\n".join(
            f"{m['role'].title()}: {m['content'][:300]}" for m in context[-4:]
        )

    prompt = f"""
{_teacher_persona(subject, topic)}

TASK: Answer the following student doubt.

Recent lecture context:
{ctx_str or "No prior context."}

Student's doubt: "{question}"

DOUBT RESOLUTION RULES:
1. RELEVANCE CHECK: If the doubt is not about "{topic}" in {subject}, respond:
   "This doubt is outside today's topic ({topic}). Please ask this after the
   lecture, or start a new session for that subject."
   Do NOT answer unrelated doubts.

2. If the doubt IS relevant, answer using this exact structure:

   **Direct Answer** (1-2 sentences — state the answer immediately)

   **Explanation** (3-5 sentences — explain the concept clearly with correct terminology)

   **Example** (1 concrete example that illustrates the answer)

   **Common Mistake** (1 sentence — what students often get wrong about this)

   **In short:** (one-line summary the student can memorise)

3. Use only facts you are certain about. Do not speculate.
4. Use markdown formatting.
5. End with: "Does this clear your doubt? Feel free to ask a follow-up! 💡"
"""
    model = _gemini_model(temperature=0.0)
    return model.generate_content(prompt).text.strip()


# ── SESSION SUMMARY GENERATOR ─────────────────────────────────────────────────

def ai_session_summary(subject: str, topic: str, messages: List[dict]) -> str:
    """
    Generate a structured, exam-ready summary of what was taught in the session.
    """
    # Build a clean conversation log (cap at 4000 chars)
    conversation = "\n".join(
        f"{'TEACHER' if m['role'] == 'assistant' else 'STUDENT'}: {m['content']}"
        for m in messages
    )[:4000]

    prompt = f"""
You are generating an academic study note from a completed lecture session.

Subject : {subject}
Topic   : {topic}

Lecture transcript:
{conversation}

Generate a structured, exam-ready study summary using EXACTLY this format:

## {topic} — Lecture Summary

### Key Concepts Covered
(Bullet list of every concept that was taught, one line each)

### Detailed Notes
(For each concept: a 2-3 sentence explanation using correct terminology)

### Examples Discussed
(List any examples, analogies, or problems that came up during the lecture)

### Important Definitions
(Any technical terms defined during the lecture, formatted as Term: definition)

### Common Mistakes to Avoid
(Mistakes or misconceptions that were highlighted)

### What to Study Next
(1-2 logical follow-up topics based on what was covered today)

RULES:
- Only include content that was actually discussed in the transcript.
- Do not add external information not present in the conversation.
- Use academic language appropriate for {subject}.
- Keep total length under 400 words.
"""
    model = _gemini_model(temperature=0.0)
    return model.generate_content(prompt).text.strip()


# ╔══════════════════════════════════════════════════════════╗
# ║  5. PYDANTIC SCHEMAS                                     ║
# ╚══════════════════════════════════════════════════════════╝

# ── Auth ──────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    name: str

# ── Syllabus ──────────────────────────────────────────────
class SyllabusCreate(BaseModel):
    subject: str
    topics: List[str]
    exam_date: Optional[datetime] = None

class SyllabusOut(BaseModel):
    id: int
    subject: str
    topics: List[str]
    total_topics: int
    exam_date: Optional[datetime]
    created_at: datetime
    class Config:
        from_attributes = True

# ── Study Plan ────────────────────────────────────────────
class PlanGenerateRequest(BaseModel):
    syllabus_id: int
    start_date: Optional[datetime] = None
    daily_hours: int = 2

class PlanOut(BaseModel):
    id: int
    syllabus_id: int
    schedule: Any
    total_days: int
    summary: Optional[str]
    generated_at: datetime
    class Config:
        from_attributes = True

# ── Lecture ───────────────────────────────────────────────
class LectureOut(BaseModel):
    id: int
    topic: str
    scheduled_date: datetime
    status: str
    lecture_content: Optional[str]
    class Config:
        from_attributes = True

# ── Lecture Session ───────────────────────────────────────
class StartSessionRequest(BaseModel):
    lecture_id: int
    mode: str = "text"   # text | voice

class SessionOut(BaseModel):
    id: int
    lecture_id: int
    mode: str
    is_active: bool
    started_at: datetime
    progress_pct: float
    messages: Any
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    session_id: int
    message: str

class ChatResponse(BaseModel):
    session_id: int
    reply: str
    progress_pct: float

class EndSessionRequest(BaseModel):
    session_id: int
    progress_pct: float = 100.0

# ── Doubt ─────────────────────────────────────────────────
class DoubtRequest(BaseModel):
    session_id: int
    question: str

class DoubtOut(BaseModel):
    id: int
    question: str
    answer: str
    asked_at: datetime
    resolved: bool
    class Config:
        from_attributes = True

# ── Calendar ──────────────────────────────────────────────
class CalendarEventOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    event_date: datetime
    event_type: str
    lecture_id: Optional[int]
    class Config:
        from_attributes = True

# ── Alarm ─────────────────────────────────────────────────
class AlarmCreate(BaseModel):
    lecture_id: Optional[int] = None
    label: str
    alarm_time: datetime
    minutes_before: int = 10

class AlarmOut(BaseModel):
    id: int
    label: str
    alarm_time: datetime
    minutes_before: int
    is_active: bool
    lecture_id: Optional[int]
    class Config:
        from_attributes = True

# ── Progress ──────────────────────────────────────────────
class ProgressOut(BaseModel):
    total_lectures: int
    completed_lectures: int
    total_doubts: int
    resolved_doubts: int
    streak_days: int
    completion_pct: float
    last_active: Optional[datetime]
    class Config:
        from_attributes = True


# ╔══════════════════════════════════════════════════════════╗
# ║  6. APP INIT                                             ║
# ╚══════════════════════════════════════════════════════════╝

app = FastAPI(
    title="EduPlatform API",
    description="AI-powered education backend — lectures, study plans, calendar & doubt solving",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)


# ╔══════════════════════════════════════════════════════════╗
# ║  7. ROUTES — AUTH                                        ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new student account."""
    if db.query(UserModel).filter(UserModel.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = UserModel(
        name=req.name,
        email=req.email,
        hashed_password=hash_password(req.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    # Init progress row
    db.add(ProgressModel(user_id=user.id))
    db.commit()
    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token, user_id=user.id, name=user.name)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login with email + password."""
    user = db.query(UserModel).filter(UserModel.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token, user_id=user.id, name=user.name)


@app.get("/auth/me", tags=["Auth"])
def me(current_user: UserModel = Depends(get_current_user)):
    """Get current logged-in user details."""
    return {"id": current_user.id, "name": current_user.name, "email": current_user.email,
            "created_at": current_user.created_at}


# ╔══════════════════════════════════════════════════════════╗
# ║  8. ROUTES — SYLLABUS                                    ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/syllabus/upload", response_model=SyllabusOut, tags=["Syllabus"])
def upload_syllabus(req: SyllabusCreate,
                    current_user: UserModel = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """
    Upload a syllabus — provide subject, list of topics, and optional exam date.
    The AI study planner will use this to build a personalized schedule.
    """
    if not req.topics:
        raise HTTPException(status_code=400, detail="Topics list cannot be empty")
    syllabus = SyllabusModel(
        user_id=current_user.id,
        subject=req.subject,
        topics=req.topics,
        total_topics=len(req.topics),
        exam_date=req.exam_date,
    )
    db.add(syllabus)
    db.commit()
    db.refresh(syllabus)
    return syllabus


@app.get("/syllabus", response_model=List[SyllabusOut], tags=["Syllabus"])
def list_syllabi(current_user: UserModel = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """List all syllabi uploaded by the current user."""
    return db.query(SyllabusModel).filter(
        SyllabusModel.user_id == current_user.id
    ).order_by(SyllabusModel.created_at.desc()).all()


@app.get("/syllabus/{syllabus_id}", response_model=SyllabusOut, tags=["Syllabus"])
def get_syllabus(syllabus_id: int,
                 current_user: UserModel = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """Get a specific syllabus by ID."""
    s = db.query(SyllabusModel).filter(
        SyllabusModel.id == syllabus_id,
        SyllabusModel.user_id == current_user.id,
    ).first()
    if not s:
        raise HTTPException(status_code=404, detail="Syllabus not found")
    return s


# ╔══════════════════════════════════════════════════════════╗
# ║  9. ROUTES — STUDY PLAN                                  ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/plan/generate", response_model=PlanOut, tags=["Study Plan"])
def generate_plan(req: PlanGenerateRequest,
                  current_user: UserModel = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """
    Generate an AI-powered study plan for a syllabus.
    Creates lectures + calendar events automatically.
    """
    syllabus = db.query(SyllabusModel).filter(
        SyllabusModel.id == req.syllabus_id,
        SyllabusModel.user_id == current_user.id,
    ).first()
    if not syllabus:
        raise HTTPException(status_code=404, detail="Syllabus not found")

    # Delete old plan if regenerating
    old_plan = db.query(StudyPlanModel).filter(
        StudyPlanModel.syllabus_id == syllabus.id
    ).first()
    if old_plan:
        db.delete(old_plan)
        db.commit()

    start = req.start_date or datetime.utcnow()
    try:
        ai_plan = ai_generate_study_plan(
            subject=syllabus.subject,
            topics=syllabus.topics,
            start_date=start,
            exam_date=syllabus.exam_date,
            daily_hours=req.daily_hours,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    plan = StudyPlanModel(
        syllabus_id=syllabus.id,
        schedule=ai_plan.get("schedule", []),
        total_days=ai_plan.get("total_days", 0),
        summary=ai_plan.get("summary", ""),
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    # Create Lecture + CalendarEvent rows for each scheduled item
    total_lectures = 0
    for item in ai_plan.get("schedule", []):
        item_type = item.get("type", "lecture")
        try:
            item_date = datetime.strptime(item["date"], "%Y-%m-%d")
        except (KeyError, ValueError):
            continue

        event_title = item.get("topic", "Study Session")

        if item_type == "lecture":
            lecture = LectureModel(
                plan_id=plan.id,
                topic=item.get("topic", ""),
                scheduled_date=item_date,
                status="pending",
            )
            db.add(lecture)
            db.commit()
            db.refresh(lecture)
            total_lectures += 1

            # Calendar event linked to lecture
            cal = CalendarEventModel(
                plan_id=plan.id,
                lecture_id=lecture.id,
                title=f"📚 {event_title}",
                description=item.get("notes", ""),
                event_date=item_date,
                event_type="lecture",
            )
        else:
            cal = CalendarEventModel(
                plan_id=plan.id,
                lecture_id=None,
                title=f"{'📝 Revision' if item_type == 'revision' else '🎯 Exam'}: {event_title}",
                description=item.get("notes", ""),
                event_date=item_date,
                event_type=item_type,
            )
        db.add(cal)

    db.commit()

    # Update progress total
    prog = db.query(ProgressModel).filter(ProgressModel.user_id == current_user.id).first()
    if prog:
        prog.total_lectures = total_lectures
        db.commit()

    return plan


@app.get("/plan/{syllabus_id}", response_model=PlanOut, tags=["Study Plan"])
def get_plan(syllabus_id: int,
             current_user: UserModel = Depends(get_current_user),
             db: Session = Depends(get_db)):
    """Get the study plan for a syllabus."""
    syllabus = db.query(SyllabusModel).filter(
        SyllabusModel.id == syllabus_id,
        SyllabusModel.user_id == current_user.id,
    ).first()
    if not syllabus:
        raise HTTPException(status_code=404, detail="Syllabus not found")
    plan = db.query(StudyPlanModel).filter(
        StudyPlanModel.syllabus_id == syllabus_id
    ).first()
    if not plan:
        raise HTTPException(status_code=404, detail="No plan generated yet. Call /plan/generate first.")
    return plan


@app.get("/plan/{plan_id}/lectures", response_model=List[LectureOut], tags=["Study Plan"])
def list_lectures(plan_id: int,
                  current_user: UserModel = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """List all lectures in a study plan."""
    plan = db.query(StudyPlanModel).filter(StudyPlanModel.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    return db.query(LectureModel).filter(
        LectureModel.plan_id == plan_id
    ).order_by(LectureModel.scheduled_date).all()


# ╔══════════════════════════════════════════════════════════╗
# ║  10. ROUTES — LECTURE ENGINE                             ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/lecture/start", response_model=SessionOut, tags=["Lecture"])
def start_lecture(req: StartSessionRequest,
                  current_user: UserModel = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    """
    Start an AI lecture session for a topic.
    Returns session with the AI's opening lecture message.
    """
    lecture = db.query(LectureModel).filter(LectureModel.id == req.lecture_id).first()
    if not lecture:
        raise HTTPException(status_code=404, detail="Lecture not found")

    # Close any existing active session for this lecture
    old = db.query(LectureSessionModel).filter(
        LectureSessionModel.user_id == current_user.id,
        LectureSessionModel.lecture_id == req.lecture_id,
        LectureSessionModel.is_active == True,
    ).first()
    if old:
        old.is_active = False
        old.ended_at = datetime.utcnow()
        db.commit()

    # Get subject from plan -> syllabus
    plan    = db.query(StudyPlanModel).filter(StudyPlanModel.id == lecture.plan_id).first()
    syllabus = db.query(SyllabusModel).filter(SyllabusModel.id == plan.syllabus_id).first() if plan else None
    subject  = syllabus.subject if syllabus else "General"

    # Generate AI intro
    try:
        intro = ai_lecture_intro(subject=subject, topic=lecture.topic)
    except Exception as e:
        intro = f"Welcome to today's lecture on **{lecture.topic}**! Let's get started. 🎓"

    # Save intro to lecture record
    lecture.status = "active"
    lecture.lecture_content = intro
    db.commit()

    # Create session with intro as first AI message
    session = LectureSessionModel(
        user_id=current_user.id,
        lecture_id=req.lecture_id,
        mode=req.mode,
        messages=[{"role": "assistant", "content": intro, "timestamp": datetime.utcnow().isoformat()}],
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.post("/lecture/chat", response_model=ChatResponse, tags=["Lecture"])
def lecture_chat_endpoint(req: ChatRequest,
                          current_user: UserModel = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    """
    Send a message during an active lecture session.
    AI responds as the teacher, continuing the lecture.
    Works for both text and voice modes.
    """
    session = db.query(LectureSessionModel).filter(
        LectureSessionModel.id == req.session_id,
        LectureSessionModel.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session already ended. Start a new session.")

    lecture  = db.query(LectureModel).filter(LectureModel.id == session.lecture_id).first()
    plan     = db.query(StudyPlanModel).filter(StudyPlanModel.id == lecture.plan_id).first()
    syllabus = db.query(SyllabusModel).filter(SyllabusModel.id == plan.syllabus_id).first() if plan else None
    subject  = syllabus.subject if syllabus else "General"

    # Append user message
    msgs = list(session.messages or [])
    msgs.append({"role": "user", "content": req.message, "timestamp": datetime.utcnow().isoformat()})

    try:
        reply = ai_lecture_chat(
            subject=subject,
            topic=lecture.topic,
            history=msgs[:-1],      # history before current message
            user_message=req.message,
            mode=session.mode,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    # Append AI reply
    msgs.append({"role": "assistant", "content": reply, "timestamp": datetime.utcnow().isoformat()})
    session.messages = msgs

    # Estimate progress based on message count (rough heuristic)
    progress = min(95.0, len([m for m in msgs if m["role"] == "user"]) * 10.0)
    session.progress_pct = progress
    db.commit()

    return ChatResponse(session_id=session.id, reply=reply, progress_pct=progress)


@app.post("/lecture/end", tags=["Lecture"])
def end_lecture(req: EndSessionRequest,
                current_user: UserModel = Depends(get_current_user),
                db: Session = Depends(get_db)):
    """
    End a lecture session. Generates a session summary and marks lecture complete.
    """
    session = db.query(LectureSessionModel).filter(
        LectureSessionModel.id == req.session_id,
        LectureSessionModel.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    lecture  = db.query(LectureModel).filter(LectureModel.id == session.lecture_id).first()
    plan     = db.query(StudyPlanModel).filter(StudyPlanModel.id == lecture.plan_id).first()
    syllabus = db.query(SyllabusModel).filter(SyllabusModel.id == plan.syllabus_id).first() if plan else None
    subject  = syllabus.subject if syllabus else "General"

    # Generate summary
    try:
        summary = ai_session_summary(subject, lecture.topic, list(session.messages or []))
    except Exception:
        summary = "Session completed successfully."

    # Mark session done
    session.is_active    = False
    session.ended_at     = datetime.utcnow()
    session.progress_pct = req.progress_pct
    db.commit()

    # Mark lecture completed
    lecture.status = "completed"
    db.commit()

    # Update progress tracker
    prog = db.query(ProgressModel).filter(ProgressModel.user_id == current_user.id).first()
    if prog:
        prog.completed_lectures += 1
        prog.last_active = datetime.utcnow()
        db.commit()

    return {"session_id": session.id, "summary": summary, "status": "completed"}


@app.get("/lecture/session/{session_id}", response_model=SessionOut, tags=["Lecture"])
def get_session(session_id: int,
                current_user: UserModel = Depends(get_current_user),
                db: Session = Depends(get_db)):
    """Get full details of a lecture session including all messages."""
    session = db.query(LectureSessionModel).filter(
        LectureSessionModel.id == session_id,
        LectureSessionModel.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ╔══════════════════════════════════════════════════════════╗
# ║  11. ROUTES — DOUBT SOLVER                               ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/doubt/ask", response_model=DoubtOut, tags=["Doubt Solver"])
def ask_doubt(req: DoubtRequest,
              current_user: UserModel = Depends(get_current_user),
              db: Session = Depends(get_db)):
    """
    Ask a doubt during or after a lecture session.
    AI gives a detailed explanation with examples.
    """
    session = db.query(LectureSessionModel).filter(
        LectureSessionModel.id == req.session_id,
        LectureSessionModel.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    lecture  = db.query(LectureModel).filter(LectureModel.id == session.lecture_id).first()
    plan     = db.query(StudyPlanModel).filter(StudyPlanModel.id == lecture.plan_id).first()
    syllabus = db.query(SyllabusModel).filter(SyllabusModel.id == plan.syllabus_id).first() if plan else None
    subject  = syllabus.subject if syllabus else "General"

    try:
        answer = ai_solve_doubt(
            subject=subject,
            topic=lecture.topic,
            question=req.question,
            context=list(session.messages or [])[-6:],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    doubt = DoubtModel(
        session_id=req.session_id,
        question=req.question,
        answer=answer,
        resolved=True,
    )
    db.add(doubt)
    db.commit()
    db.refresh(doubt)

    # Update progress
    prog = db.query(ProgressModel).filter(ProgressModel.user_id == current_user.id).first()
    if prog:
        prog.total_doubts    += 1
        prog.resolved_doubts += 1
        db.commit()

    return doubt


@app.get("/doubt/session/{session_id}", response_model=List[DoubtOut], tags=["Doubt Solver"])
def get_session_doubts(session_id: int,
                       current_user: UserModel = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    """Get all doubts asked in a session."""
    session = db.query(LectureSessionModel).filter(
        LectureSessionModel.id == session_id,
        LectureSessionModel.user_id == current_user.id,
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return db.query(DoubtModel).filter(
        DoubtModel.session_id == session_id
    ).order_by(DoubtModel.asked_at).all()


# ╔══════════════════════════════════════════════════════════╗
# ║  12. ROUTES — CALENDAR                                   ║
# ╚══════════════════════════════════════════════════════════╝

@app.get("/calendar/{plan_id}", response_model=List[CalendarEventOut], tags=["Calendar"])
def get_calendar(plan_id: int,
                 month: Optional[int] = None,
                 year: Optional[int] = None,
                 current_user: UserModel = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """
    Get all calendar events for a study plan.
    Optionally filter by month and year (e.g. ?month=6&year=2025).
    Use this to power the calendar view in your Android app.
    """
    plan = db.query(StudyPlanModel).filter(StudyPlanModel.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    q = db.query(CalendarEventModel).filter(CalendarEventModel.plan_id == plan_id)
    if month and year:
        start_of_month = datetime(year, month, 1)
        end_month = month + 1 if month < 12 else 1
        end_year  = year if month < 12 else year + 1
        end_of_month = datetime(end_year, end_month, 1)
        q = q.filter(
            CalendarEventModel.event_date >= start_of_month,
            CalendarEventModel.event_date < end_of_month,
        )
    return q.order_by(CalendarEventModel.event_date).all()


@app.get("/calendar/{plan_id}/today", response_model=List[CalendarEventOut], tags=["Calendar"])
def get_today_events(plan_id: int,
                     current_user: UserModel = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    """Get today's scheduled events — handy for the home screen widget."""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end   = today_start + timedelta(days=1)
    return db.query(CalendarEventModel).filter(
        CalendarEventModel.plan_id == plan_id,
        CalendarEventModel.event_date >= today_start,
        CalendarEventModel.event_date < today_end,
    ).order_by(CalendarEventModel.event_date).all()


@app.get("/calendar/{plan_id}/upcoming", response_model=List[CalendarEventOut], tags=["Calendar"])
def get_upcoming_events(plan_id: int,
                        days: int = 7,
                        current_user: UserModel = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Get upcoming events for the next N days (default 7)."""
    now = datetime.utcnow()
    end = now + timedelta(days=days)
    return db.query(CalendarEventModel).filter(
        CalendarEventModel.plan_id == plan_id,
        CalendarEventModel.event_date >= now,
        CalendarEventModel.event_date <= end,
    ).order_by(CalendarEventModel.event_date).all()


# ╔══════════════════════════════════════════════════════════╗
# ║  13. ROUTES — ALARMS                                     ║
# ╚══════════════════════════════════════════════════════════╝

@app.post("/alarm/set", response_model=AlarmOut, tags=["Alarms"])
def set_alarm(req: AlarmCreate,
              current_user: UserModel = Depends(get_current_user),
              db: Session = Depends(get_db)):
    """
    Set an alarm for a lecture.
    Your Android app should poll /alarm/pending to fetch due alarms and
    trigger a local notification.
    """
    alarm = AlarmModel(
        user_id=current_user.id,
        lecture_id=req.lecture_id,
        label=req.label,
        alarm_time=req.alarm_time,
        minutes_before=req.minutes_before,
    )
    db.add(alarm)
    db.commit()
    db.refresh(alarm)
    return alarm


@app.get("/alarm/list", response_model=List[AlarmOut], tags=["Alarms"])
def list_alarms(current_user: UserModel = Depends(get_current_user),
                db: Session = Depends(get_db)):
    """List all active alarms for the user."""
    return db.query(AlarmModel).filter(
        AlarmModel.user_id == current_user.id,
        AlarmModel.is_active == True,
    ).order_by(AlarmModel.alarm_time).all()


@app.get("/alarm/pending", response_model=List[AlarmOut], tags=["Alarms"])
def get_pending_alarms(current_user: UserModel = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    """
    Get alarms due in the next 15 minutes.
    Poll this endpoint every minute from the Android app to trigger local notifications.
    """
    now = datetime.utcnow()
    window = now + timedelta(minutes=15)
    return db.query(AlarmModel).filter(
        AlarmModel.user_id == current_user.id,
        AlarmModel.is_active == True,
        AlarmModel.alarm_time >= now,
        AlarmModel.alarm_time <= window,
    ).all()


@app.delete("/alarm/{alarm_id}", tags=["Alarms"])
def delete_alarm(alarm_id: int,
                 current_user: UserModel = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """Delete / dismiss an alarm."""
    alarm = db.query(AlarmModel).filter(
        AlarmModel.id == alarm_id,
        AlarmModel.user_id == current_user.id,
    ).first()
    if not alarm:
        raise HTTPException(status_code=404, detail="Alarm not found")
    db.delete(alarm)
    db.commit()
    return {"detail": "Alarm deleted"}


# ╔══════════════════════════════════════════════════════════╗
# ║  14. ROUTES — PROGRESS                                   ║
# ╚══════════════════════════════════════════════════════════╝

@app.get("/progress", response_model=ProgressOut, tags=["Progress"])
def get_progress(current_user: UserModel = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """Get the student's overall learning progress stats."""
    prog = db.query(ProgressModel).filter(
        ProgressModel.user_id == current_user.id
    ).first()
    if not prog:
        raise HTTPException(status_code=404, detail="Progress record not found")
    completion = (
        (prog.completed_lectures / prog.total_lectures * 100)
        if prog.total_lectures > 0 else 0.0
    )
    return ProgressOut(
        total_lectures=prog.total_lectures,
        completed_lectures=prog.completed_lectures,
        total_doubts=prog.total_doubts,
        resolved_doubts=prog.resolved_doubts,
        streak_days=prog.streak_days,
        completion_pct=round(completion, 1),
        last_active=prog.last_active,
    )


@app.get("/progress/history/{plan_id}", tags=["Progress"])
def get_lecture_history(plan_id: int,
                        current_user: UserModel = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Get completion status of every lecture in a plan."""
    lectures = db.query(LectureModel).filter(
        LectureModel.plan_id == plan_id
    ).order_by(LectureModel.scheduled_date).all()

    return [
        {
            "lecture_id": l.id,
            "topic": l.topic,
            "scheduled_date": l.scheduled_date,
            "status": l.status,
        }
        for l in lectures
    ]


# ╔══════════════════════════════════════════════════════════╗
# ║  15. HEALTH CHECK                                        ║
# ╚══════════════════════════════════════════════════════════╝

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "EduPlatform API", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}