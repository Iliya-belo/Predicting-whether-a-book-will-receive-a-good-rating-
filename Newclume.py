import pandas as pd
import json
import numpy as np


df  = pd.read_csv("other_finaly_option1.csv")
is_successful = [1 if i >=3.8 else 0 for i in df.rating]
print(is_successful)
df['is_successful'] = is_successful
df2 = df.copy()
df2.to_csv('finaly result logestic.csv')