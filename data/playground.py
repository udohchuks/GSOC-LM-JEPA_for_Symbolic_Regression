import pandas as pd
import numpy as np
import os
path_df = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
df = pd.read_csv(f'{path_df}/FeynmanEquations.csv')
row = df.iloc[10]

print("Equation:", row['Formula'])
print("Filename:", row['Filename'])
print("Num vars:", row['# variables'])
print("v1_name:", row['v1_name'])
print()

# Load the actual data file
path  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'Feynman_with_units')
data = np.loadtxt(f"{path}/{row['Filename']}")
print("Data shape:", data.shape)
print("First 3 rows:")
print(data[:3])
print()

print("Last column is target (y), rest are inputs (X)")
print("X shape:", data[:, :-1].shape)
print("y shape:", data[:, -1].shape)