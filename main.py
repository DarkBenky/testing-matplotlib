import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pprint as pp

WINDOW = 20

# Download data
btc = yf.download('BTC-USD', period='1d', interval='5m')

def create_df(data, window=20):
    x = []
    y = []
    # Calculate price changes
    change = data['Close'].pct_change().values
    values = data.values
    
    def vwap(data):
        v = data['Volume'].values
        p = data['Close'].values
        return np.sum(v * p) / np.sum(v)
    
    # def rsi(data):
    #     delta = data['Close'].diff()
    #     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    #     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    #     rs = gain / loss
    #     return 100 - (100 / (1 + rs))
    
    # def macd(data):
    #     short = data['Close'].ewm(span=12, adjust=False).mean()
    #     long = data['Close'].ewm(span=26, adjust=False).mean()
    #     return short - long
    
    def atr(data):
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        tr = np.maximum(high - low, np.maximum(high - close, close - low))
        return np.mean(tr)
    
    def obv(data):
        close = data['Close'].values
        volume = data['Volume'].values
        return np.where(close > close[-1], volume, np.where(close < close[-1], -volume, 0)).sum()
    
    def adx(data):
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        tr = np.maximum(high - low, np.maximum(high - close, close - low))
        tr = np.mean(tr)
        return tr
        
    
    def POC_profile(data):
        poc = {}
        prices = data['Close'].values.tolist()
        volumes = data['Volume'].values.tolist()
        for i in range(0, len(prices)):
            if poc.get(prices[i][0]):
                poc[prices[i][0]] += volumes[i][0]
            else:
                poc[prices[i][0]] = volumes[i][0]
        return max(poc, key=poc.get)
    
    def TPO_profile(data):
        tpo = {}
        prices = data['Close'].values.tolist()
        for i in range(0, len(prices)):
            if tpo.get(prices[i][0]):
                tpo[prices[i][0]] += 1
            else:
                tpo[prices[i][0]] = 1
        return max(tpo, key=tpo.get)
    
    
    for i in range(0, len(values)-window):
        x_ = values[i:i+window]

        v = vwap(data.iloc[i:i+window])
        # x_ = np.append(x_, rsi(data.iloc[i:i+window]))
        # x_ = np.append(x_, macd(data.iloc[i:i+window]))
        a = atr(data.iloc[i:i+window])
        o = obv(data.iloc[i:i+window])
        ad = adx(data.iloc[i:i+window])
        p = POC_profile(data.iloc[i:i+window])
        t = TPO_profile(data.iloc[i:i+window])

        features = [v, a, o, ad, p, t]
        #Normalize features
        features = (features - np.mean(features)) / np.std(features) 

        # Normalize data
        x_ = (x_ - x_.mean()) / x_.std()

        x_ = x_.tolist()
        x_.append(features)
        x_ = np.array(x_)

        y_ = change[i+window] if i+window < len(change) else 0
        x.append(x_)
        y.append(y_)
    
    return np.array(x), np.array(y)

# Create features and labels
x, y = create_df(btc, window=WINDOW)

def print_data(data):
    data = data.tolist()
    for i in range(len(data)):
        for j in range(len(data[i])):
            print(f"{data[i][j]:.5f}", end=' | ')
        print("\n")
    print("\n")

print(x.shape, y.shape, "Data shape")
print_data(x[50])
print_data(x[51])
print_data(x[52])

# Show the first 50 x values as a heatmap
plt.figure(figsize=(10, 5))
plt.imshow(x.reshape(-1, x.shape[0])[:3], cmap='hot', interpolation='nearest', aspect='auto')
plt.colorbar()
plt.title('Heatmap of x Values')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.show()

# Show the first 50 y values as a line plot
plt.figure(figsize=(10, 5))
plt.plot(y[:50], label='y values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('First 50 y Values')
plt.legend()
plt.show()
