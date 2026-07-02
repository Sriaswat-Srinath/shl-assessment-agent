import os
from dotenv import load_dotenv

# 1. Force load .env
load_dotenv()

# 2. DEBUG: Print the status of the key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ CRITICAL ERROR: GROQ_API_KEY is missing! Check your .env file.")
else:
    print(f"✅ SUCCESS: Loaded GROQ_API_KEY starting with: {api_key[:10]}...")

# 3. Import everything else AFTER the key is verified
from fastapi import FastAPI
from app.models import ChatRequest, ChatResponse
from app.agent import AgentOrchestrator
from app.retrieval import validate_url_exists

app = FastAPI(title="SHL Assessment Agent")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_turns = len([m for m in request.messages if m.role == "user"])
    if user_turns >= 8:
        return ChatResponse(
            reply="Maximum conversation limit reached. Please finalize your selection.",
            recommendations=[],
            end_of_conversation=True
        )

    orchestrator = AgentOrchestrator(request.messages)
    result = orchestrator.run()
    
    validated_recs = []
    for rec in result["recommendations"]:
        if not validate_url_exists(rec.url):
            print(f"ERROR: Blocked hallucinated URL: {rec.url}")
            continue 
        validated_recs.append(rec)

    return ChatResponse(
        reply=result["reply"],
        recommendations=validated_recs,
        end_of_conversation=result["end_of_conversation"]
    )