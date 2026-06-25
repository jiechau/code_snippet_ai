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

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Split the data into features (X) and target (y)
X = df[['watermelon', 'apple', 'grape']].values
y = df['cost'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the neural network model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = RegressionModel()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Print the training loss every epoch
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')


# Evaluate the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

    print(f'Test Loss: {test_loss/len(test_loader):.4f}')

# Use the trained model to make predictions on new data
watermelon_qty = np.random.randint(1, QTY_RANGE)
apple_qty = np.random.randint(1, QTY_RANGE)
grape_qty = np.random.randint(1, QTY_RANGE)
exact = (watermelon_qty * watermelon_cost) + (apple_qty * apple_cost) + (grape_qty * grape_cost)
new_data = torch.tensor([[watermelon_qty, apple_qty, grape_qty]], dtype=torch.float32)
prediction = model(new_data).item()
print(f"{watermelon_qty} {apple_qty} {grape_qty} = {exact}, predict: {prediction:.0f}")
