import numpy as np

nInst = 100
currentPos = np.zeros(nInst)

import tensorflow as tf
import pandas as pd
import time

# Position limit
POSLIM = 10000

# Load model
# Expected input shape: (None, 25, 100)
# Output shape: (None, 25, 100)
model = tf.keras.models.load_model('dense_model')

normalize_params = pd.read_csv("data_normalization.csv")
mean = normalize_params['mean'].to_numpy()
std = normalize_params['std'].to_numpy()

def normalize_prices(prices):
    prices = (prices - mean) / std
    return prices

def unnormalize_prices(prices):
    prices = (prices * std) + mean
    return prices

def getMyPosition(prcSoFar):
    start = time.perf_counter()
    global currentPos

    day_no = prcSoFar.shape[1]

    if day_no < 25:
        return currentPos

    # Obtain last prices
    last_prices = prcSoFar[:,-25:].T
    last_prices = tf.expand_dims(last_prices, 0)

    # Today's prices
    price = prcSoFar[:,-1].T
    
    # Predicted prices
    predictions = unnormalize_prices(model.predict(normalize_prices(last_prices)))

    # Predicted tomorrow's prices
    next_prices = predictions[0][-1]

    # Difference of tomorrow's and today's
    diff_prices = next_prices - price

    # Decide new position based off change
    currentPos = currentPos + np.sign(diff_prices) * POSLIM // next_prices

    end = time.perf_counter()
    # print(f"Elapsed time: {end - start} seconds")
    return currentPos

    
