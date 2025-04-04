from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .detector import check_plagiarism

app = FastAPI()

class CodeRequest(BaseModel):
    code: str

@app.post("/check")
def check_code(request: CodeRequest):
    try:
        verdict, references = check_plagiarism(request.code)
        return {"verdict": verdict, "references": references}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
