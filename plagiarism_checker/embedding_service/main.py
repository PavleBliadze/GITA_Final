from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import get_embedding

app = FastAPI()  

class EmbedRequest(BaseModel):
    code: str

@app.post("/embed")
def embed_code(req: EmbedRequest):
    try:
        embedding = get_embedding(req.code)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
