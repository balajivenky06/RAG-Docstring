
import pandas as pd

file_path = 'results/Consolidated.xlsx'
# Read with header at row 4 (0-indexed)
df = pd.read_excel(file_path, header=4)
print("Columns:", df.columns.tolist())
print(df.head())
