# necessary imports
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from keras.models import Model
from keras.layers import TextVectorization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import EarlyStopping
import yaml
import pickle
import logging
from utils import *

# Configure logging (file + console)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler('training.log')
fh.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

def log_summary(line):
    logging.info(line)  # Sends model.summary() lines to logger

# read configs
configs = yaml.safe_load(open("config.yaml"))
data_dir = configs["data_dir"] # get data directory

# read in training data of fake and real news
fake_df = pd.read_csv(f"{data_dir}/Fake.csv").assign(target = 1)
true_df = pd.read_csv(f"{data_dir}/True.csv").assign(target = 0)
data_df = pd.concat([fake_df,true_df])

# convert date to datetime
data_df['date'] = pd.to_datetime(data_df['date'], format='mixed', errors='coerce')
# get rid of rows where the date wasn't valid
data_df = data_df.dropna(subset=["date"])
# get the year, month, day, and day of the week
data_df['year'] = data_df['date'].dt.year
data_df['month'] = data_df['date'].dt.month
data_df['day'] = data_df['date'].dt.day
data_df['day_of_week'] = data_df['date'].dt.dayofweek

# split into training, validation, and test sets
# since this has a time stamp we use time to split
# percentiles used loaded from config
train_cutoff = configs["train_cutoff"]
val_cutoff = configs["val_cutoff"]

train_end = data_df.date.quantile(train_cutoff)
val_end = data_df.date.quantile(val_cutoff)

train_df = data_df[data_df.date <= train_end]
val_df = data_df[(data_df.date > train_end) & ( data_df.date <= val_end)]
test_df = data_df[data_df.date > val_end]

# preprocess all text columns 
# leaving this out for now because training seems to converge faster without preprocessing
#for col in configs["text_cols"]:
#    train_df[col] = train_df[col].apply(preprocess)
#    val_df[col] = val_df[col].apply(preprocess)
#    test_df[col] = test_df[col].apply(preprocess)
# tokenize all the text columns

# create one column from the training data that combines the text from the text columns.
train_df = train_df.assign(
    combined = train_df[configs["text_cols"]].agg(" ".join, axis=1)
)
vocab_vectorizer = TextVectorization(output_mode='int')
vocab_vectorizer.adapt(train_df["combined"].to_list())
vocab = vocab_vectorizer.get_vocabulary()
vocab_size = len(vocab)
# create a max_lengths dictionary based on each column
max_lens = {}
for col in configs["text_cols"]:
    maxlen = train_df[col].apply(lambda x: len(x)).max()
    max_lens[col] = min(maxlen, 1000)
# create empty list for sequences for train, val, test
sequences_train = []
sequences_val = []
sequences_test = []

for col in configs["text_cols"]:
    # we only fit the tokenizer on the training data
    # the other data uses the trained tokenizer
    seq_train = tokenize(train_df[col], vocab, max_len=max_lens[col])
    seq_val = tokenize(val_df[col], vocab, max_len=max_lens[col])
    seq_test = tokenize(test_df[col], vocab, max_len=max_lens[col])
    sequences_train.append(seq_train)
    sequences_val.append(seq_val)
    sequences_test.append(seq_test)

date_features = ['month', 'day', 'day_of_week']


# Extract Date Features
train_date_features = train_df[
    date_features
    ].values.astype(np.float32)

val_date_features = val_df[
    date_features
    ].values.astype(np.float32)
test_date_features = test_df[
    date_features
].values.astype(np.float32)

# Normalize Date Features
max_vals = np.max(train_date_features, axis=0)
train_date_features /= max_vals  # Min-max normalization
val_date_features /= max_vals    # use train max to avoid leakage
test_date_features /= max_vals   # use train max to avoid leakage


# Convert tokenized sequences to NumPy arrays
X1_train, X2_train, X3_train = [np.array(seq) for seq in sequences_train]
X1_val, X2_val, X3_val = [np.array(seq) for seq in sequences_val]
X1_test, X2_test, X3_test = [np.array(seq) for seq in sequences_test]

print (max([len(x) for x in X1_val]))