#reading news data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np

file = 'C:/Users/Anuj/Documents/PERSONAL/Projects/Charles/data/all_news.csv'

df = pd.read_csv(file, encoding='Windows-1252')

df_rates = df.Sentiment

def get_score(rate):
    if rate == "negative": return -1
    if rate == "neutral": return 0
    if rate == "positive": return 1
    return 

df_score = [get_score(i) for i in df_rates]

# neutral_score = 0
# positive_score = 0
# negative_score = 0

# for score in df_score:
#     if score == 0: neutral_score +=1
#     if score == 1: positive_score +=1
#     if score == -1: negative_score +=1

# plt.bar(["Positive", "Neutral", "Negative"], [positive_score, neutral_score, negative_score])
# plt.show()

MODEL_NAME = 'bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

token_lens = []

for txt in df.Review:
    tokens = tokenizer.encode(txt)
    token_lens.append(len(tokens))

MAX_LEN = 100


