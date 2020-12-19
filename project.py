import pandas as pd
import pandas_datareader as dr
#%matplotlib inline


#load data from yahoo finance
df= dr.data.get_data_yahoo('BAC', start='2006-12-05', end='2020-12-05')
df.columns = ['High',' Low','Open','Close','Volume','AdjClose']
#df.index.name = 'date'
print(df.head(5))
# save to file

df.to_csv('stock.csv')