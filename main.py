import os
import json
import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# ─── Bootstrap ────────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Pine AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    # Restrict to your frontend origin in production, e.g. ["http://localhost:3000"]
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)

# ─── Models ───────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str          # "user" | "assistant"
    content: str       # plain text — images are handled via image_data below

class ChatRequest(BaseModel):
    message: str
    hasImage: bool = False
    image_data: Optional[str] = None          # base64 data-URL, e.g. "data:image/png;base64,..."
    conversation_history: list[HistoryMessage] = []   # previous turns for multi-turn context

# ─── System prompt (evaluated once at startup so the date is accurate) ────────

SYSTEM_PROMPT = f"""
You are Pine AI, a professional technical analyst with a helpful, human-like personality.
The current date is {datetime.date.today().strftime("%B %d, %Y")}.

GUIDELINES:
1. If the user greets you or asks general questions (who are you, how are you, what date is it),
   respond naturally and warmly as a supportive peer.
2. Only provide a 'signal' when the user supplies a chart, asks for a specific price analysis,
   or explicitly requests a trade setup.
3. For general conversation set "signal" to null.
4. Use **bold** for emphasis in your analysis.
5. Keep responses concise and actionable.

You MUST return ONLY a valid JSON object — no markdown fences, no preamble:
{{
  "analysis": "Your natural response or technical breakdown here.",
  "signal": {{
    "action": "LONG or SHORT or NEUTRAL",
    "entry": "$Price",
    "stopLoss": "$Price",
    "takeProfit": "$Price",
    "confidence": 85,
    "riskReward": "1:2",
    "timeframe": "1H"
  }}
}}

Set "signal" to null (JSON null, not the string "null") when no trade setup is warranted.
"""

# ─── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_chart(request: ChatRequest):
    try:
        # ── Build message list ──────────────────────────────────────────────
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in request.conversation_history:
            if turn.role in ("user", "assistant"):
                messages.append({"role": turn.role, "content": turn.content})

        # ── Build the current user turn ────────────────────────────────────
        if request.hasImage and request.image_data:
            # Groq expects the full data-URL: "data:<mime>;base64,<data>"
            # Validate it looks correct before sending
            if not request.image_data.startswith("data:image"):
                raise HTTPException(status_code=400, detail="Invalid image format. Must be a base64 data-URL.")

            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": request.image_data},
                },
                {
                    "type": "text",
                    "text": request.message or "Analyse this chart and suggest a trade setup.",
                },
            ]
            # Use vision model for image requests
            model = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            user_content = request.message
            model = "llama-3.3-70b-versatile"

        messages.append({"role": "user", "content": user_content})

        # ── Call Groq ──────────────────────────────────────────────────────
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=1024,
        )

        raw = completion.choices[0].message.content

        # ── Parse & validate ───────────────────────────────────────────────
        try:
            ai_response = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=502, detail="Model returned invalid JSON.")

        if "analysis" not in ai_response:
            raise HTTPException(status_code=502, detail="Model response missing 'analysis' field.")

        return ai_response

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        print(f"[Pine AI] {type(e).__name__}: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "Pine AI v1.0"}