'''
a watermelon costs $100. an apple costs $10. a grape costs $1.
1個西瓜100元。1個蘋果10元。一顆葡萄1元。
created a DataFrame with 100 rows of data, where each row contains random quantities of watermelons, apples, and grapes, along with the calculated total cost. 
創建一個包含 100 行數據的數據框架,其中每一行都包含隨機數量的西瓜、蘋果和葡萄,以及計算出的總成本。
'''

DATA_NUM = 10_000
QTY_RANGE = 10
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 32

import pandas as pd
import numpy as np

# Define the cost of each item
watermelon_cost = 100
apple_cost = 10
grape_cost = 1
# Create an empty list to store the data
data = []
# Generate 100 rows of data
for _ in range(DATA_NUM):
    # Generate random quantities for each item (between 1 and QTY_RANGE)
    watermelon_qty = np.random.randint(1, QTY_RANGE)
    apple_qty = np.random.randint(1, QTY_RANGE)
    grape_qty = np.random.randint(1, QTY_RANGE)
    # Calculate the total cost
    total_cost = (watermelon_qty * watermelon_cost) + (apple_qty * apple_cost) + (grape_qty * grape_cost)
    # Append the data to the list
    data.append([watermelon_qty, apple_qty, grape_qty, total_cost])
# Create the DataFrame
df = pd.DataFrame(data, columns=['watermelon', 'apple', 'grape', 'cost'])


'''
Use this existing df as the training dataset to train a regression model.
利用這個現有的 df 當做訓練數據集。訓練一個回歸的模型。
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Split the data into features (X) and target (y)
X = df[['watermelon', 'apple', 'grape']].values
y = df['cost'].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[3]),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), 
          verbose=1)


'''
Use the trained model to make predictions on new data.
使用訓練好的模型進行新數據的預測。
'''
import random
watermelon_qty = random.randint(1, QTY_RANGE)
apple_qty = random.randint(1, QTY_RANGE)
grape_qty = random.randint(1, QTY_RANGE)
exact = (watermelon_qty * watermelon_cost) + (apple_qty * apple_cost) + (grape_qty * grape_cost)

new_data = np.array([[watermelon_qty, apple_qty, grape_qty]])  # Example input: 3 watermelons, 5 apples, 2 grapes
prediction = model.predict(new_data)
print(f"{watermelon_qty} {apple_qty} {grape_qty} = {exact}, predict:{prediction[0][0]:.0f}")

