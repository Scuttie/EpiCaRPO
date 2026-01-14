import pandas as pd

# 조사할 파일 경로 (예: math_train_path)
file_path = "/mnt/chatbot30TB/jewonyeom/verl_qwen3/verl/data/aime2025/test.parquet"

df = pd.read_parquet(file_path)
print("--- Columns ---")
print(df.columns)
print("\n--- Data Sample (First Row) ---")
print(df.iloc[0].to_dict())