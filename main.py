from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("resona-base-pilot")

# Connect to Hugging Face's free Inference API
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/clip-ViT-B-32"

class MoodRequest(BaseModel):
    mood: str

@app.get("/")
def health_check():
    return {"status": "Resona API is live and routing!"}

@app.post("/discover")
def discover_items(req: MoodRequest):
    # 1. Ask Hugging Face to translate the text to math for free
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    hf_response = requests.post(API_URL, headers=headers, json={"inputs": req.mood})
    
    if hf_response.status_code != 200:
        return {"error": "AI routing currently warming up, please try again in 10 seconds."}
        
    mood_vector = hf_response.json()
    
    # Safely format the vector math for Pinecone
    if isinstance(mood_vector, list) and isinstance(mood_vector[0], list):
        mood_vector = mood_vector[0]
    
    # 2. Query Pinecone
    db_response = index.query(
        vector=mood_vector,
        top_k=2, 
        include_metadata=True
    )
    
    results = [match["metadata"] for match in db_response["matches"]]
    return {"recommendations": results}
