#!/usr/bin/env python
# coding: utf-8

# Import statements

# In[2]:


# import statements
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model.model import predict
import json
import numpy as np
import pickle
import datetime



# TODO:  write the API endpoints.  
# YOUR CODE GOES HERE

app = FastAPI()

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    prediction: float

@app.get("/")
def home():
    return {"API Test": "API is Operational"}

@app.post("/predict/delays", response_model = PredictionOut)
def get_predict(Order, AirportEnc, DepTime, ArrTime):
    data = [Order, AirportEnc, DepTime, ArrTime]
    prediction = predict(data)
    return {prediction}

