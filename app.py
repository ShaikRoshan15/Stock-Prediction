import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Predefined list of stock symbols
stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
print("Available stocks:", stock_list)
stock_symbol = input("Enter stock symbol from the list: ").upper()

if stock_symbol not in stock_list:
    print("Invalid stock symbol. Please choose from the list.")
    exit()

# Generate synthetic stock price data
def generate_synthetic_data(days=1000):
    np.random.seed(hash(stock_symbol) % (2**32))  # Seed based on stock symbol
    x = np.arange(days)
    prices = 50 + 5 * np.sin(x * 0.02) + np.random.normal(0, 1, days)
    return prices

# Generate data
stock_prices = generate_synthetic_data()

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_prices.reshape(-1, 1))

time_step = 60
X, Y = [], []
for i in range(len(scaled_data) - time_step - 1):
    X.append(scaled_data[i:(i + time_step), 0])
    Y.append(scaled_data[i + time_step, 0])

X = np.array(X).reshape(len(X), time_step, 1)
Y = np.array(Y)

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
Y_train = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=3, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Take the last output
        return out

# Train model
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Prepare test data
test_data = scaled_data[-120:]
X_test, Y_test = [], []
for i in range(len(test_data) - time_step - 1):
    X_test.append(test_data[i:(i + time_step), 0])
    Y_test.append(test_data[i + time_step, 0])

X_test = torch.tensor(np.array(X_test).reshape(len(X_test), time_step, 1), dtype=torch.float32)

# Predict stock values
with torch.no_grad():
    predictions = model(X_test)
    predictions = scaler.inverse_transform(predictions.numpy())

actual_prices = scaler.inverse_transform(np.array(Y_test).reshape(-1, 1))

if predictions[-1] > predictions[0]:
    decision = f"Suggestion: Buy {stock_symbol} (Upward trend detected)"
else:
    decision = f"Suggestion: Leave {stock_symbol} (Downward trend detected)"

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(actual_prices, label=f'Actual {stock_symbol}', color='blue')
plt.plot(predictions, label=f'Predicted {stock_symbol}', color='red', linestyle='dashed', linewidth=2, alpha=0.7)
plt.title(f'Stock Price Prediction for {stock_symbol}')
plt.legend()
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.grid()
plt.show()

print(decision)
