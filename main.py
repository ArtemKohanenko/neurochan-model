from fastapi import FastAPI
import torch
import numpy as np
import transformers
from transformers import pipeline
from pydantic import BaseModel


app = FastAPI()

pipe = pipeline("text-generation", model="tinkoff-ai/ruDialoGPT-small")

class Request(BaseModel):
    text: str
    context: str

@app.post("/predict")
async def predict(req: Request):
    input = f'@@ПЕРВЫЙ@@{req.context}@@ВТОРОЙ@@{req.text}@@ПЕРВЫЙ@@'
    result = pipe(input, max_new_tokens=70, do_sample=True)[0]['generated_text']
    print(result)
    result_splited = result.split('@@ПЕРВЫЙ@@')
    pred = result_splited[2].split('@@ВТОРОЙ@@')[0]
    return {"data": { "predict": pred } }

#uvicorn main:app --reload