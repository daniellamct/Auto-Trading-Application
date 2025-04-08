from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

task = 'sentiment-latest'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
model_dir = f'News_Sentiment_Analysis/cardiffnlp/twitter-roberta-base-{task}'


if not os.path.exists(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
else:
    print(f"The folder '{model_dir}' already exists. Skipping the model loading and saving.")
