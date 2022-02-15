from tokenize import String
from PredictionModel import PredictionModel
import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from transformers import pipeline
import itertools
from operator import itemgetter
from fastapi import FastAPI
from pydantic import BaseModel

# Set up prediction model
model_folder_path = "jasminejwebb/KeywordIdentifier"

load_model = AutoModelForTokenClassification.from_pretrained(model_folder_path, num_labels=4)
load_tokenizer = AutoTokenizer.from_pretrained(model_folder_path)

model = PredictionModel(verbose = False)
model.createTokenizerAndModel(model_folder_path)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/keywords")
async def identify_keywords(input_text: str):
    kw = model.predict(sentence = input_text)
    return {"dataframe": kw}