import pandas as pd

dt = pd.read_csv("../data/raw/iris.csv")
print(dt)
print(dt.iloc[0:10, 0:2])

dt.to_csv("../data/processed/iris_processed.csv", index=False)



