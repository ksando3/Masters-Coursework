#!/usr/bin/env python
# coding: utf-8

# Output of cleaned data file should look like this:
# 
# | YEAR | MONTH | DAY | DAY_OF_WEEK | ORG_AIRPORT | DEST_AIRPORT | SCHEDULED_DEPARTURE | DEPARTURE_TIME | DEPARTURE_DELAY | SCHEDULED_ARRIVAL | ARRIVAL_TIME | ARRIVAL_DELAY |
# |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
# | integer | integer | integer | integer | string | string | integer | integer | integer | integer | integer | integer |
# 
# and should be named "imported_data" and saved as a .csv file

# In[1]:


import pandas as pd
import logging


# In[2]:


imported_data = pd.read_csv("C:/Users/cfman/OneDrive/Desktop/WGUClasses/D602 Deployment/Task 2/Data/T_ONTIME_REPORTING.csv")


# Renaming the columns to match what is shown in the poly_regressor file.

# In[3]:


imported_data = imported_data.rename(columns = {"DAY_OF_MONTH": "DAY", "ORIGIN": "ORG_AIRPORT", "DEST": "DEST_AIRPORT",
                               "CRS_DEP_TIME": "SCHEDULED_DEPARTURE", "DEP_TIME": "DEPARTURE_TIME", "DEP_DELAY": "DEPARTURE_DELAY", 
                               "CRS_ARR_TIME": "SCHEDULED_ARRIVAL", "ARR_TIME": "ARRIVAL_TIME", "ARR_DELAY": "ARRIVAL_DELAY"})


# In[4]:


## Check if column names are correct

#imported_data.head()


# We can see that the column names line up with what is needed for the other file. Now that that is cleaned up, we can move on to Part C.

# In[6]:


## https://docs.python.org/3/howto/logging.html

imported_data.to_csv('C:/Users/cfman/OneDrive/Desktop/WGUClasses/D602 Deployment/Task 2/Data/imported_data.csv', index = False)

# configure logger
logname = "imported_data.txt"
logging.basicConfig(filename=logname,
                    filemode='w',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.info("Imported Data Exported to CSV Log")

