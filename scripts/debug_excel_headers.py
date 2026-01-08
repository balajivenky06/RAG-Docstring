
import pandas as pd

file_path = 'results/Consolidated.xlsx'
df = pd.read_excel(file_path, header=None, nrows=10)
print(df)
