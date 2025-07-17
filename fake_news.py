# silence warnings from TF
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
# necessary imports
import pandas as pd
import numpy as np
import yaml
import pickle
import logging

from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from keras.models import Model
from keras.layers import TextVectorization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import EarlyStopping

from utils import *

def setup_logging(log_file='training.log'):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        # set level to info
        logger.setLevel(logging.INFO)
        # create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        # create streaming handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and set both logs to formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
# Configure logging (file + console)
logger = setup_logging()

def log_summary(line):
    logging.info(line)  # Sends model.summary() lines to logger

# read configs
configs = yaml.safe_load(open("config.yaml"))
data_dir = configs["data_dir"] # get data directory
fake_path = os.path.join(data_dir, "Fake.csv")
true_path = os.path.join(data_dir, "True.csv")
# read in training data of fake and real news
fake_df = pd.read_csv(fake_path).assign(target = 1)
true_df = pd.read_csv(true_path).assign(target = 0)
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
    maxlen = train_df[col].str.len().max()
    max_lens[col] = min(maxlen, 1000)
# create empty list for sequences for train, val, test
X_train = {}
X_val = {}
X_test = {}

for col in configs["text_cols"]:
    # we only fit the tokenizer on the training data
    # the other data uses the trained tokenizer
    seq_train = tokenize(train_df[col], vocab, max_len=max_lens[col])
    seq_val = tokenize(val_df[col], vocab, max_len=max_lens[col])
    seq_test = tokenize(test_df[col], vocab, max_len=max_lens[col])
    X_train[col] = seq_train
    X_val[col] = seq_val
    X_test[col] = seq_test

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
# replace any max of 0 with 1
max_vals[max_vals == 0] = 1
train_date_features /= max_vals  # Min-max normalization
val_date_features /= max_vals    # use train max to avoid leakage
test_date_features /= max_vals   # use train max to avoid leakage
X_train["input_date"] = train_date_features
X_val["input_date"] = val_date_features
X_test["input_date"] = test_date_features

# Model Parameters

embedding_dim = configs["embedding_dim"]  # Size of embedding vectors
lstm_units = configs["lstm_units"]    # Number of LSTM units

# Layers of the model
def create_model(
        text_cols: list[str] = configs["text_cols"],
        vocab_size: int = vocab_size,
        date_features: list[str] = date_features,
        embedding_dim: int = embedding_dim,
        lstm_units:int =lstm_units
) -> Model:
    """
    creates and compiled Keras model

    Parameters
    ----------
    text_cols : List[string]
        list of text column names
    vocab_size : Int
        size of vocabulary from TextVectorization
    date_features : List[String]
        list of date feature column names
    embedding_dim : Int
        dimension for embedding text data
    lstm_units : Int
        number of LSTMs to put in LSTM layer
    
    Returns
    -------
    model : Model
        compiled model with inputs text columns and date features
        and output the classification
    """
    # create input layers
    input_layers = {
        col: Input(shape=(max_lens[col],), name=col) for col in text_cols
        }
    input_layers["input_date"] = Input(shape=(len(date_features),), name="input_date")
    # create embedding layers
    embed_layers = {
        col: Embedding(input_dim=vocab_size, output_dim=embedding_dim, name = f'embed_{col}')(input_layers[col]) 
        for col in text_cols
    }
    # create lstm_layers
    lstm_layers = {
            col: LSTM(lstm_units, name=f'lstm_{col}')(embed_layers[col]) for col in text_cols
    }
    # create dense layer for date fields
    date_dense = Dense(16, activation='relu', name='date_dense')(input_layers["input_date"])
    # concatenate LSTMs and date dense layer
    concat = Concatenate(name="concatenate")([*lstm_layers.values(), date_dense])
    # create output layer
    output = Dense(1, activation='sigmoid', name='output')(concat)
    # create the model
    model = Model(
        inputs=input_layers,
        outputs=output
    )
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

model.summary(print_fn=log_summary)

# set up the output for the training, validation, and test sets
y_train = train_df["target"]
y_val = val_df["target"]
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
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data = (X_val, y_val),
    callbacks=[early_stopping],
    epochs=10
    )

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

# save the model to the Model directory
# get the path where the models should be saved
model_dir = configs["model_dir"]
home_dir = os.path.expanduser("~")
model_path = os.path.join(home_dir,model_dir)
# make sure the model path exists or create it
os.makedirs(output_dir, exist_ok=True)
model.save(f"{model_path}/fake_news_model.keras")
# Save tokenizer to the Model directory
with open(f"{model_path}/tokenizer.pkl", "wb") as f:
    pickle.dump(vocab_vectorizer, f)
