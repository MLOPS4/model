from cleaning import clean_data
from train import retrain

df = clean_data("1765341306.csv")

retrain(df, "v1")