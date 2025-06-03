#!/usr/bin/env python
# coding: utf-8

# Import statements

# In[2]:


# import statements
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model import predict
import json
import numpy as np
import pickle
import datetime


# Opening arrival airport list

# In[3]:


# Opening airport encodings
f = open('C:/Users/cfman/OneDrive/Desktop/WGUClasses/D602 Deployment/Task 3/airport_encodings.json')
# returns JSON object as a dictionary
airports = json.load(f)


# In[3]:


#airports


# In[4]:


def create_airport_encoding(airport: str, airports: dict) -> np.array:
    """
    create_airport_encoding is a function that creates an array the length of all arrival airports from the chosen
    departure aiport.  The array consists of all zeros except for the specified arrival airport, which is a 1. Required
    as part of input to saved polynomial regression model.

    Parameters
    ----------
    airport : str
        The specified arrival airport code as a string.

    airports: dict
        A dictionary containing all of the arrival airport codes served from the chosen departure airport.

    Returns
    -------
    np.array
        A NumPy array the length of the number of arrival airports.  All zeros except for a single 1 
        denoting the arrival airport.  Returns None if arrival airport is not found in the input list.

    """
    temp = np.zeros(len(airports))
    if airport in airports:
        temp[airports.get(airport)] = 1
        temp = temp.T
        return temp
    else:
        return None


# In[55]:


# TODO:  write the back-end logic to provide a prediction given the inputs
# requires finalized_model.pkl to be loaded
# the model must be passed a NumPy array consisting of the following:
# (polynomial order, encoded airport array, departure time as seconds since midnight, arrival time as seconds since midnight)
# the polynomial order is 1 unless you changed it during model training in Task 2
# YOUR CODE GOES HERE

with open(f"C:/Users/cfman/OneDrive/Desktop/WGUClasses/D602 Deployment/Task 3/finalized_model.pkl", "rb") as f:
    model = pickle.load(f)
    
def predict(model, data): # Input data as a list. IE: [1, "SEA", "9:30", "15:30"]
    
    # Taking given departure and arrival times as strings and converting them to seconds since midnight.
    # Easier for the user to not have to convert themselves
    departure_time = data[2]
    arrival_time = data[3]
    departure_time = sum(x * int(t) for x, t in zip([3600, 60, 1], departure_time.split(":")))
    arrival_time = sum(x * int(t) for x, t in zip([3600, 60, 1], arrival_time.split(":")))
    # Setting order equal to that of the first input as required
    order = data[0]
    #print("Order: ", order)
    
    # Creating a list of lists with the order as the first entry
    edited_data = [[order]]
    #print("Edited_data : ", edited_data)
    
    # Running the encoded airport function to generate the information about a given airport. Converting the numpy array to a list
    encoded_data = create_airport_encoding(data[1], airports).tolist()
    #print("Encoded_data : ", encoded_data)
    
    # Adding the converted numpy array from the airport encoding function to the new list of lists
    edited_data.append(encoded_data)
    #print("Edited_data : ", edited_data)
    
    # flattening the list of lists to create 1 list with all necessary inputs
    edited_data = [x for xs in edited_data for x in xs]
    #print("Edited_data : ", edited_data)
    
    # Adding the departure time since midnight
    edited_data.append(departure_time)
    #print("Edited_data : ", edited_data)
    
    # adding the arrival time since midnight
    edited_data.append(arrival_time)
    #print("Edited_data : ", edited_data)
    
    # Changing the list to a numpy array as required
    edited_data = np.array([edited_data])
    #print("Edited_data : ", edited_data)
    
    # Reshaping the array as needed
    edited_data = edited_data.reshape(1,-1)
    #print("Edited_data : ", edited_data)
    
    
    # Returning the prediced delay in minutes
    prediction = model.predict(edited_data)
    print("Predicted Delay: ", round(prediction[0][0], 2), "minutes")
    #print(type(prediction))


# In[57]:


# Manual test to check if code works

predict(model, [1, "SAN", "9:30", "13:17"])



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

