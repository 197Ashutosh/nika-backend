from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from supabase import create_client, Client
import os
import datetime
from dotenv import load_dotenv
from groq import Groq
from typing import Optional

load_dotenv()
app = FastAPI(title="Nika Enterprise OS Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None

def run_system_cron():
    print(f"[NIKA CRON] Executed at {datetime.datetime.now()}. Systems nominal.")

scheduler = BackgroundScheduler()
scheduler.add_job(run_system_cron, 'interval', minutes=60)
scheduler.start()

class GoalCreate(BaseModel):
    title: str
    weight: int
    level: str
    status: str

class GoalUpdate(BaseModel):
    status: Optional[str] = None
    completion_percentage: Optional[int] = None

class FeedbackUpdate(BaseModel):
    employee_submitted: Optional[bool] = None
    employee_rating: Optional[str] = None
    manager_submitted: Optional[bool] = None
    manager_rating: Optional[str] = None
    manager_text: Optional[str] = None
    is_flagged: Optional[bool] = None
    is_soft_flag: Optional[bool] = None
    sentiment_label: Optional[str] = None

class ConfigUpdate(BaseModel):
    is_configured: bool

@app.get("/api/config")
def get_config():
    res = supabase.table("system_config").select("*").eq("id", 1).execute()
    return res.data[0] if res.data else {"is_configured": False}

@app.put("/api/config")
def update_config(cfg: ConfigUpdate):
    res = supabase.table("system_config").update({"is_configured": cfg.is_configured}).eq("id", 1).execute()
    return res.data[0]

@app.get("/api/goals")
def get_goals():
    res = supabase.table("goals").select("*").order("created_at", desc=True).execute()
    return res.data

@app.post("/api/goals")
def create_goal(goal: GoalCreate):
    res = supabase.table("goals").insert({
        "title": goal.title, "weight": goal.weight, "hierarchy_level": goal.level, "status": goal.status
    }).execute()
    return res.data[0]

@app.put("/api/goals/{goal_id}/status")
def update_goal_status(goal_id: str, payload: GoalUpdate):
    update_data = payload.dict(exclude_unset=True)
    res = supabase.table("goals").update(update_data).eq("id", goal_id).execute()
    return res.data[0]

@app.delete("/api/goals/{goal_id}")
def delete_goal(goal_id: str):
    supabase.table("goals").delete().eq("id", goal_id).execute()
    return {"status": "deleted"}

@app.get("/api/feedback")
def get_feedback():
    res = supabase.table("feedback").select("*").limit(1).execute()
    return res.data[0] if res.data else None

@app.put("/api/feedback/{feedback_id}")
def update_feedback(feedback_id: str, payload: FeedbackUpdate):
    update_data = payload.dict(exclude_unset=True)
    res = supabase.table("feedback").update(update_data).eq("id", feedback_id).execute()
    return res.data[0]

class AIGoalReq(BaseModel):
    title: str

@app.post("/api/ai/smart-goal")
def generate_smart_goal(req: AIGoalReq):
    if not groq_client:
        return {"suggestion": "CRITICAL ERROR: GROQ_API_KEY is missing from Render Environment Variables!"}
        
    try:
        prompt = f"""
        You are an elite corporate performance strategist. 
        The employee has drafted this raw, basic goal: "{req.title}"
        
        Your task: Rewrite this completely into a formal, highly professional SMART goal (Specific, Measurable, Achievable, Relevant, Time-bound). 
        DO NOT just repeat their words. You must inject corporate metrics (e.g., percentages, Q3 deadlines, efficiency metrics) to make it sound professional.
        
        Respond ONLY with the final 1-sentence rewritten goal. No introductory text, no quotes.
        """
        
        res = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict data processor. Only output the final rewritten string."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",  # <--- UPDATED TO GROQ'S NEWEST ACTIVE MODEL
            temperature=0.4
        )
        return {"suggestion": res.choices[0].message.content.strip(' "')}
    except Exception as e:
        return {"suggestion": f"API FAILED. The exact error is: {str(e)}"}

class AISentimentReq(BaseModel):
    comment: str

@app.post("/api/ai/sentiment")
def analyze_sentiment(req: AISentimentReq):
    if groq_client:
        try:
            prompt = f"""You are an advanced HR Sentiment AI. Analyze this employee review for sentiment. 
            Pay close attention to negations (e.g., 'not good' is NEGATIVE, 'not great' is NEGATIVE).
            Respond with EXACTLY ONE WORD: 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.
            Review Text: "{req.comment}"
            """
            res = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], 
                model="llama-3.1-8b-instant", # <--- UPDATED HERE TOO
                temperature=0.0
            )
            sentiment = res.choices[0].message.content.strip().upper()
            if "POSITIVE" in sentiment: sentiment = "POSITIVE"
            elif "NEGATIVE" in sentiment: sentiment = "NEGATIVE"
            else: sentiment = "NEUTRAL"
            return {"sentiment": sentiment, "is_flagged": sentiment == "NEGATIVE"}
        except Exception:
            pass

    # Fallback Lexicon Engine
    text = req.comment.lower()
    if "not good" in text or "not great" in text or "not well" in text: return {"sentiment": "NEGATIVE", "is_flagged": True}
    if "not bad" in text or "not terrible" in text: return {"sentiment": "POSITIVE", "is_flagged": False}
    
    bad_words = ["quit", "hate", "terrible", "worst", "bad", "poor", "awful", "unacceptable"]
    good_words = ["good", "great", "excellent", "awesome", "best", "love", "outstanding", "exceeds"]
    if any(word in text for word in bad_words): return {"sentiment": "NEGATIVE", "is_flagged": True}
    elif any(word in text for word in good_words): return {"sentiment": "POSITIVE", "is_flagged": False}
    else: return {"sentiment": "NEUTRAL", "is_flagged": False}
