import pandas as pd
import json

df = pd.read_parquet('data/serving/oof_predictions.parquet')
print('OOF columns:', df.columns.tolist())
print('OOF shape:', df.shape)
print(df.head(2))

with open('data/serving/labels.json') as f:
    print('Labels JSON:', f.read())
