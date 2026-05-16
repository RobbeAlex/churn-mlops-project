import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

os.makedirs('data/raw', exist_ok=True)

# Path bajo data/raw
file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Carga dataset de Kaggle y lo guarda como CSV local
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "blastchar/telco-customer-churn",
  file_path
)
# Guarda el archivo en la ruta esperada por tus tests
df.to_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", index=False)
