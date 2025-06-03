#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import logging
import numpy as np
import datetime


# In[2]:


imported_data = pd.read_csv("C:/Users/cfman/OneDrive/Desktop/WGUClasses/D602 Deployment/Task 2/Data/imported_data.csv")


# In[3]:


# configure logger
logname = "exported_data.txt"
logging.basicConfig(filename=logname,
                    filemode='w',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.info("Exporting Cleaned Data to CSV Log")


# In[4]:


#imported_data.head()


# Creating a new dataframe that only looks at the departures from my chosen airport, LAX.

# In[5]:


cleaned_data = imported_data[imported_data['ORG_AIRPORT'] == "LAX"]
logging.info("Filtering to LAX successful.")
#cleaned_data.head()


# In[6]:


cleaned_data = cleaned_data.reset_index(drop=True)


# In[7]:


cleaned_data.head()


# Going to check for any duplicated occurrences as part of the data cleaning.

# In[8]:


#print(cleaned_data.shape)


duplicates = cleaned_data.duplicated(keep = False)
#duplicates.value_counts()


# No duplicated values are found, so we can now move on to checking for missing values.

# In[9]:


cleaned_data.isna().sum()


# It is worth noting that DEPARTURE_TIME and ARRIVAL_TIME are listed as integers but actually represent a real life time. Thus no integer will have the 10s place be greater than 5. If we were to just subtract the scheduled departure time and actual departure time through python, we would obtain an incorrect result of the difference. Similiarly with the scheduled arrival and actual arrival. Because of this, it is not appropriate to look at the distributions for those variables. The xx_DELAY variables account for the correct difference in time.

# I will choose to REMOVE the NA values in both DEPARTURE_TIME and ARRIVAL_TIME and then recheck to see if there are still missing values.

# In[10]:


cleaned_data = cleaned_data.dropna(subset=['DEPARTURE_TIME', 'ARRIVAL_TIME'])
logging.info("Dropping NA values from Departure_Time and Arrival_Time successful.")


# In[11]:


cleaned_data.isna().sum()


# I will convert the float columns without missing values to integers now.

# In[12]:


cleaned_data = cleaned_data.astype({"DEPARTURE_TIME": 'int', "DEPARTURE_DELAY": 'int', "ARRIVAL_TIME": 'int'})
logging.info("Changing float columns without missing values to ints successful.")


# In[13]:


missing_Arrival = cleaned_data[cleaned_data["ARRIVAL_DELAY"].isnull()]


# In[14]:


indices = missing_Arrival.index.tolist()
scheduled = list(missing_Arrival["SCHEDULED_ARRIVAL"])
scheduled = [str(x) for x in scheduled]

for i in range(0,len(scheduled)):
    if len(scheduled[i]) < 3:
        scheduled[i] = '00' + scheduled[i]
#print(scheduled)

scheduled_edited = []

for t in scheduled:
    scheduled_edited.append(datetime.datetime.strptime(t,'%H%M').strftime('%H:%M'))
scheduled_edited = [str(x) for x in scheduled_edited]
#print (scheduled_edited)

arrival = list(missing_Arrival["ARRIVAL_TIME"])
arrival = [str(x) for x in arrival]

for i in range(0,len(arrival)):
    if len(arrival[i]) < 3:
        arrival[i] = '00' + arrival[i]
#print(arrival)

arrival_edited = []

for t in arrival:
    arrival_edited.append(datetime.datetime.strptime(t,'%H%M').strftime('%H:%M'))
arrival_edited = [str(x) for x in arrival_edited]
#print (arrival_edited)


time_difference = []
for i in range(0, len(arrival)):
    # convert time string to datetime
    scheduled_edited[i] = datetime.datetime.strptime(scheduled_edited[i], "%H:%M")
    #print('Start time:', scheduled_edited.time())

    arrival_edited[i] = datetime.datetime.strptime(arrival_edited[i], "%H:%M")
    #print('End time:', arrival_edited.time())

    # get difference
    delta = arrival_edited[i] - scheduled_edited[i]

    # time difference in seconds
    #print(f"Time difference is {delta.total_seconds()} seconds") 
   
    # time difference in minutes
    mins = delta.total_seconds() / 60
    
    # In some instances the scheudled arrival is say 14:04 but they arrived at 3:47. This should be a delay of about 13.5 hours,
    # but the current value shows the flight arrived about 10.5 hours early. Clearly that is unreasonable and would never happen
    # so I'm going to reverse the calculation if the delta suggests the flight is early by more than 2 hours  
    
    if mins < -120.0:
        mins = 1440 + mins
    time_difference.append(mins)
    x = indices[i]
    logging.info("Calculated Arrival_Delay for index %s ", x)

#print(time_difference)


# In[15]:


# [In text citation: Dr. Middleton, K (n.d) Getting Started with D206 Data Types, Distributions, and Univariate Imputation]
#plt.hist(cleaned_data["ARRIVAL_DELAY"])
plt.show()

# Check the statistics of the data before editing
#print(cleaned_data["ARRIVAL_DELAY"].describe())
#print("Median: ", cleaned_data["ARRIVAL_DELAY"].median())


# Going to replace the missing values with the proper calculated arrival delays.

# In[16]:


# [In text citation: Dr. Middleton, K (n.d) Getting Started with D206 Data Types, Distributions, and Univariate Imputation]
cleaned_data.loc[cleaned_data['ARRIVAL_DELAY'].isna(), 'ARRIVAL_DELAY'] = time_difference
logging.info("Calculated Arrival Delay times imputed successfully.")
# Check the statistics of the data after editing
#print(cleaned_data["ARRIVAL_DELAY"].describe())
#print("Median: ", cleaned_data["ARRIVAL_DELAY"].median())

#print(cleaned_data["ARRIVAL_DELAY"].isna().sum())

#plt.hist(cleaned_data["ARRIVAL_DELAY"])
plt.show()


# In[17]:


#cleaned_data.isna().sum()


# As another data cleaning check, I will check the departure and arrival delays for any outliers

# In[18]:


#boxplot_population = seaborn.boxplot(x = "DEPARTURE_DELAY", data = cleaned_data)


# In[19]:


#cleaned_data['DEPARTURE_DELAY'].describe()


# In[20]:


#cleaned_data[cleaned_data["DEPARTURE_DELAY"] > 400].count()


# It appears that the largest outliers are past 400 so I will exclude them

# In[21]:


## Removing the  instances outlier and rechecking the plot
DEPARTURE_DELAY_outliers = cleaned_data[ (cleaned_data["DEPARTURE_DELAY"] > 400)]
#Income_outliers.info()

cleaned_data.drop(cleaned_data[ (cleaned_data["DEPARTURE_DELAY"] > 400)].index, inplace = True )
DEPARTURE_DELAY_outliers = seaborn.boxplot(x = "DEPARTURE_DELAY", data = cleaned_data)
logging.info("Outliers dropped successfully.")


# I am going to retain the rest of these outliers as I think any more outlier removal could result in signficant data loss. 

# In[22]:


#boxplot_population = seaborn.boxplot(x = "ARRIVAL_DELAY", data = cleaned_data)


# Since departure delay and arrival delay are closely linked together, I am going to leave the outliers in ARRIVAL_DELAY as is to maintain data.

# Now that the NA values have been dealt with, I can convert the the ARRIVAL_DELAY permanently to integers as specified in the poly_regressor file.

# In[23]:


cleaned_data = cleaned_data.astype({"ARRIVAL_DELAY": 'int'})
logging.info("Arrival Delay converted to int successfully.")


# In[24]:


#display(cleaned_data.dtypes)


# In[25]:


#cleaned_data.head()


# In[26]:


cleaned_data.to_csv('C:/Users/cfman/OneDrive/Desktop/WGUClasses/D602 Deployment/Task 2/Data/cleaned_data.csv', index = False)
logging.info("Cleaned_data exported successfully.")

