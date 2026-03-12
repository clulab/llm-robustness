import pandas as pd
import json

df = pd.read_json('eval_output.json', lines=True)

def parse_row(val):
    if isinstance(val, str):
        return json.loads(val)
    return val

df = df[0].apply(parse_row).apply(pd.Series)
df.to_csv('results.csv', index=False)