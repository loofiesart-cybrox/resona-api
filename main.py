from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
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

print("Loading the AI Brain for text translation...")
model = SentenceTransformer('clip-ViT-B-32')

class MoodRequest(BaseModel):
    mood: str

@app.get("/")
def health_check():
    return {"status": "Resona API is live and routing!"}

@app.post("/discover")
def discover_items(req: MoodRequest):
    mood_vector = model.encode(req.mood).tolist()
    
    response = index.query(
        vector=mood_vector,
        top_k=2, 
        include_metadata=True
    )
    
    results = [match["metadata"] for match in response["matches"]]
    return {"recommendations": results}
