# necessary imports
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
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
configs = yaml.safe_load(open("config_2.yaml"))
data_dir = configs["data_dir"] # get data directoryy
model_dir = configs["model_dir"]
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
# we load the tokenizer from the model trained with the full data since
# we are using transfer learning

with open(f"{model_dir}/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
max_lens = {}
for col in configs["text_cols"]:
    maxlen = train_df[col].apply(lambda x: len(x)).max()
    max_lens[col] = min(maxlen, 1000)

sequences_train = []
sequences_val = []
sequences_test = []
for col in configs["text_cols"]:
    # we only fit the tokenizer on the training data
    # the other data uses the trained tokenizer
    seq_train = tokenize(train_df[col], tokenizer, max_len=max_lens[col])
    seq_val = tokenize(val_df[col], tokenizer, max_len=max_lens[col])
    seq_test = tokenize(test_df[col], tokenizer, max_len=max_lens[col])
    sequences_train.append(seq_train)
    sequences_val.append(seq_val)
    sequences_test.append(seq_test)


# Extract Date Features
train_date_features = train_df[
    [ 'month', 'day', 'day_of_week']
    ].values.astype(np.float32)

val_date_features = val_df[
    ['month', 'day', 'day_of_week']
    ].values.astype(np.float32)
test_date_features = test_df[
    [ 'month', 'day', 'day_of_week']
].values.astype(np.float32)

# Normalize Date Features
max_vals = np.max(train_date_features, axis=0)
train_date_features /= max_vals  # Min-max normalization
val_date_features /= max_vals    # use train max to avoid leakage
test_date_features /= max_vals   # use train max to avoid leakage


# Convert tokenized sequences to NumPy arrays
X1_train, X2_train = [np.array(seq) for seq in sequences_train]
X1_val, X2_val = [np.array(seq) for seq in sequences_val]
X1_test, X2_test = [np.array(seq) for seq in sequences_test]

# Model Parameters
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding index
embedding_dim = configs["embedding_dim"]  # Size of embedding vectors
lstm_units = configs["lstm_units"]    # Number of LSTM units

# Layers of the model

# Input layers
input1 = Input(shape=(max_lens["title"],), name="title")
input2 = Input(shape=(max_lens["text"],), name="text")
# input3 = Input(shape=(max_lens["subject"]), name="subject")
input_date = Input(shape=(3,), name="input_date")

# Embedding Layers
embed1 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed1')(input1)
embed2 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed2')(input2)
# embed3 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed3')(input3)

# LSTM Layers
lstm1 = LSTM(lstm_units, name='lstm1')(embed1)
lstm2 = LSTM(lstm_units, name='lstm2')(embed2)
# lstm3 = LSTM(lstm_units, name='lstm3')(embed3)

# Dense Layer for Date Features
date_dense = Dense(16, activation='relu', name='date_dense')(input_date)

# Concatenate All Features
concat = Concatenate(name='concatenate')([lstm1, lstm2, date_dense])

# Fully Connected Layers
output = Dense(1, activation='sigmoid', name='output')(concat)

# compile the model
model = Model(inputs=[input1, input2, input_date], outputs=output)

# get weights from model that includes the summary
model_full = load_model(f"{model_dir}/fake_news_model.h5")

for layer in model.layers:
    try:
        full_layer = model_full.get_layer(name=layer.name)
        layer.set_weights(full_layer.get_weights())
        print(f"✅ Transferred weights for layer: {layer.name}")
    except (ValueError, AttributeError):
        print(f"⚠️ Skipped layer: {layer.name}")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary(print_fn=log_summary)

# set up the inputs/output for the training, validation, and test sets
X_train = [X1_train, X2_train, train_date_features]
y_train = train_df["target"]
X_val = [X1_val, X2_val, val_date_features]
y_val = val_df["target"]
X_test = [X1_test, X2_test, test_date_features]
y_test = test_df["target"]

# set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    patience=3,          # Number of epochs with no improvement after which training will be stopped
    min_delta=0.001,     # Minimum change in the monitored quantity to qualify as an improvement
    mode='min',          # 'min' for metrics that should decrease (like loss), 'max' for metrics that should increase (like accuracy)
    verbose=1,           # Verbosity mode (0 for silent, 1 for messages)
    restore_best_weights=True # Restore model weights from the epoch with the best monitored value
    )

# train the model
history = model.fit(X_train,y_train,validation_data = (X_val, y_val), callbacks=[early_stopping], epochs=10)

# add the history to the log

log_training_history( history, logging)

# save history as a json as well
# Save to JSON
save_training_history_json(history, "history.json")

# look at performance on the test set.
# get predictions
y_pred = model.predict(X_test)

# convert to 0 or 1
y_pred_vals = (y_pred >= 0.5).reshape((len(y_pred),)).astype("int")

logging.info("Confusion matrix for test set")
logging.info("-----------------------------")
logging.info(confusion_matrix(y_test, y_pred_vals))

logging.info(f"Accuracy score: {accuracy_score(y_test,y_pred_vals)}")


model.save(f"{model_dir}/fake_news_model_2.h5")

# Save tokenizer to the Model directory
with open(f"{model_dir}/tokenizer_2.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

