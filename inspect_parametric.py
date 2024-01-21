import pandas as pd

parameters = pd.read_csv("test/resources/subset_parametric.csv", index_col="Unnamed: 0")

micro = pd.read_csv("microBIKED_processed.csv")

p_columns = set(parameters.columns)
m_columns = set(micro.columns)
print(p_columns.issubset(m_columns))
print(p_columns.difference(m_columns))

