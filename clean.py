#uncomment the lines and use the required once only. i had cleaned all datasets using only this file.
import pandas as pd
import numpy as np
df=pd.read_csv("yield_df.csv")#pesticides.csv,rainfall.csv,temp.csv and the yield and yield_df.csv files are are cleaned and no null values so no need to perform any tasks on them.
# df.replace("", np.nan, inplace=True)
# df["average_rain_fall_mm_per_year"] = pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")
# df.fillna(df.mean(numeric_only=True),inplace=True)
print(df.head(),df.tail(),df.info())
print(df.isnull().sum())
# df.to_csv('temp(C).csv')
# df.to_csv('rainfall(C).csv')
# df['Value']=df['Value'].astype(int)
# print(df['Unit'].nunique())
# print(df['Domain'].unique())
# df.to_csv('pesticides(C).csv')