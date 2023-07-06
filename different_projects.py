

#region Create Sensor Capturing Toy App
import pandas as pd
from datetime import datetime
import pytz

# Read the data from the file
df = pd.read_csv("/Users/benediktjordan/Library/Mobile Documents/com~apple~CloudDocs/SmartphoneActivityApp/sensordata.csv")
df = pd.read_csv("/Users/benediktjordan/Downloads/sensordata.csv")

#convert the time column to datetime; the  "Timestamp" column is in fractional seconds since 1970-01-01 00:00:00
df["Timestamp Datetime"] = pd.to_datetime(df["Timestamp"], unit="ms")

# Create a timezone object for Berlin
tz_berlin = pytz.timezone('Europe/Berlin')

# Convert datetime to Berlin timezone
df["Timestamp Datetime"] = df["Timestamp Datetime"].dt.tz_localize(pytz.utc).dt.tz_convert(tz_berlin)

# print for every minute between the first and last timestamp the number of rows
# this is the number of samples per minute
for i in range(0, int((df["Timestamp Datetime"].max() - df["Timestamp Datetime"].min()).total_seconds() / 60)):
    print("Minute", i)
    start = df["Timestamp Datetime"].min() + pd.Timedelta(minutes=i)
    end = df["Timestamp Datetime"].min() + pd.Timedelta(minutes=i+1)
    print(start, end, len(df[(df["Timestamp Datetime"] >= start) & (df["Timestamp Datetime"] < end)]))
#endregion

#region create automatic shield text with OpenAI API
api_key = "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"
import os
import openai
openai.organization = "org-gm2mG3XuhFDDkTl5PUHklADV"
openai.api_key = api_key
openai.Model.list()

#create a completion
openai.Completion.create(
  model="ada",
  prompt="Say this is a test",
  max_tokens=2,
  temperature=0
)


#endregion