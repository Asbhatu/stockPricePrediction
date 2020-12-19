from influxdb import InfluxDBClient
import time
import datetime
import pandas as pd
client = InfluxDBClient(host='localhost', port='8086')
client.switch_database('stock')
csvReader = pd.read_csv('stock.csv')
json =[]
for row_index, row in csvReader.iterrows():
    date = row[0]
    High = row[1]
    Low = row[2]
    Open = row[3]
    Close = row[4]
    Volume = row[5]
    AdjClose = row[6]
    element = datetime.datetime.strptime(date,"%Y-%m-%d") 
    timestamp = datetime.datetime.isoformat(element)
    json_body = {
        "measurement": "stockforecast",
        "time": timestamp,
        "fields": {
                "High":High,
                "Low":Low,
                "Open":Open,
                "Close":Close,
                "Volume":Volume,
                "AdjClose":AdjClose,
        }
    }
    json.append(json_body)
client.write_points(json)
