from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.config import Config
from src.data.tokenizer import Tokenizer
from src.evaluation.translate import load_model, translate


model = None
src_tokenizer = None
tgt_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, src_tokenizer, tgt_tokenizer

    print("Loading translation model...")

    src_tokenizer = Tokenizer("tokenizers/en.model")
    tgt_tokenizer = Tokenizer("tokenizers/ru.model")
    model = load_model()

    print("Model is ready!")

    yield


app = FastAPI(
    title="English-Russian Transformer Translator",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse("static/index.html")

class TranslationRequest(BaseModel):
    text: str


class TranslationResponse(BaseModel):
    translation: str

@app.post("/translate", response_model=TranslationResponse)
def translate_text(request: TranslationRequest):
    if not request.text.strip():
        return {"translation": ""}

    result = translate(
        request.text,
        model,
        src_tokenizer,
        tgt_tokenizer
    )

    return {"translation": result}