import pandas as pd
df=pd.read_csv("pesticides.csv")
df['Value']=df['Value'].astype(int)
# print(df.head(),df.tail(),df.info())
print(df.nunique().sum(),df.uni)