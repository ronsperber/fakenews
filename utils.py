"""
utilities module
"""

import re
import tensorflow as tf
import keras
from keras.layers import TextVectorization
import json
import logging

def log_training_history(history, logger=None, level='info'):
    """
    Logs the training history from a Keras model.fit() call.

    Parameters
    ----------
    history : History
        The History object returned by model.fit().
    logger : logging.Logger, optional
        The logger instance to use. If None, uses the root logger.
    level : str
        Logging level as string ('info', 'debug', etc.)
    """
    if logger is None:
        logger = logging.getLogger()

    log_fn = getattr(logger, level.lower(), logger.info)

    log_fn("Training history:")
    for epoch in range(len(next(iter(history.history.values())))):
        log_line = f"Epoch {epoch + 1}: " + ", ".join(
            f"{metric}={history.history[metric][epoch]:.4f}"
            for metric in history.history
        )
        log_fn(log_line)


def save_training_history_json(history, filepath="history.json"):
    """
    Saves the training history to a JSON file.

    Parameters
    ----------
    history : keras.callbacks.History
        The History object returned by model.fit().
    filepath : str
        Path to the output JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(history.history, f, indent=4)

def load_training_history_json(filepath="history.json"):
    """
    Loads training history from a JSON file.

    Returns
    -------
    dict
        A dictionary of lists, same as history.history
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def preprocess(text):
    """
    Preprocesses text: lowercases, removes URLs, HTML, and extra whitespace.

    Parameters
    ----------
    text : str
        Single text input

    Returns
    -------
    str
        Cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)                 # Remove HTML
    text = re.sub(r'\s+', ' ', text).strip()           # Normalize whitespace
    return text

   

def tokenize(text_col, vocab, max_len=1000):
    """
    takes a  text column, vocabulary,  then tokenizes and pads

    Parameters
    ----------
    text_col : series
        pandas series of text to be tokenized
    vocab: vocabulary
        vocabulary to be used to transform the text
    max_len: int
        max length to be used for padding sequences
    
    Returns
    -------
    Array
        tokenized text
    """
    max_col_len = text_col.apply(lambda x:len(x)).max()
    
    vectorizer = TextVectorization(output_mode='int',
                                    output_sequence_length=max_len,
                                    vocabulary=vocab)
    padded_text_seq = vectorizer(text_col.to_list())
    return padded_text_seq
