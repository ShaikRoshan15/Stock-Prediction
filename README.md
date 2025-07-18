# 📈 LSTM-Based Stock Price Prediction (Synthetic Data)

This project is a demonstration of using **PyTorch** to build an LSTM model for predicting stock prices. Instead of real-time financial APIs, we simulate stock data for different symbols to focus on model training and forecasting using deep learning.

---

## 🚀 Features

- Synthetic stock data generation for 5 popular companies (`AAPL`, `GOOGL`, `MSFT`, `AMZN`, `TSLA`)
- LSTM model built using PyTorch
- Data scaling with MinMaxScaler
- Prediction and visualization using Matplotlib
- Trend detection with Buy/Leave suggestions
- Simple command-line interface

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **PyTorch** for LSTM modeling
- **NumPy** and **Scikit-learn** for preprocessing
- **Matplotlib** for plotting
- **Synthetic Data Generation** (based on sine wave + noise)

---

## 📂 Project Structure

```
├── app.py                # Main code to run the prediction
├── README.md             # Documentation
```

---

## 🧠 Model Architecture

- Input: 60 time steps (previous stock prices)
- Model: 3-layer LSTM with 100 hidden units and dropout
- Output: Single predicted price
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam

---

## 📊 How It Works

1. Choose a stock from the predefined list
2. Synthetic prices are generated and scaled
3. Data is split into input (X) and target (Y) sequences
4. LSTM model is trained on the sequences
5. Predictions are made on unseen (test) data
6. Results are plotted with decision advice based on trend

---

## ▶️ How to Run

1. **Install required libraries:**

```bash
pip install numpy torch scikit-learn matplotlib
```

2. **Run the app:**

```bash
python app.py
```

3. **Follow the prompt to enter a stock symbol (e.g., `AAPL`)**

---

## 💡 Example Output

```
Available stocks: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
Enter stock symbol from the list: AAPL
Epoch [1/10], Loss: 0.0032
...
Suggestion: Buy AAPL (Upward trend detected)
```

> A plot window will also open showing the predicted vs. actual prices.

---

## 📌 Notes

- This project uses **synthetic data**, not real financial data.
- It is ideal for learning and testing LSTM architectures on time series data.

---

## 👨‍💻 Author

- Developed by **[Shaik Roshan]**
- GitHub: [your-username](https://github.com/ShaikRoshan15)

---

## 📜 License

This project is open-source and free to use under the [MIT License](https://opensource.org/licenses/MIT).
