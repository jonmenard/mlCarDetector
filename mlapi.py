from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ScoringItem(BaseModel):
    YearsAtCompany: float
    EmploteeSatisfaction: float
    Position: str
    Salary: int

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
    return item