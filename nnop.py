# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For neural networks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# For the cumulative distribution function
from scipy.stats import norm

# 1. Generate synthetic options data using Black-Scholes formula
def black_scholes_call_price(S, K, T, r, sigma):
    """Calculate Black-Scholes European call option price"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put_price(S, K, T, r, sigma):
    """Calculate Black-Scholes European put option price"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def black_scholes_delta_call(S, K, T, r, sigma):
    """Calculate Delta of a European call option"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def black_scholes_gamma(S, K, T, r, sigma):
    """Calculate Gamma of a European option"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def black_scholes_theta_call(S, K, T, r, sigma):
    """Calculate Theta of a European call option"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = - r * K * np.exp(-r * T) * norm.cdf(d2)
    theta = term1 + term2
    return theta

# Generate synthetic data
np.random.seed(42)

num_samples = 10000

# Generate random inputs within reasonable ranges
S = np.random.uniform(50, 150, num_samples)        # Stock price between $50 and $150
K = np.random.uniform(50, 150, num_samples)        # Strike price between $50 and $150
T = np.random.uniform(0.01, 2, num_samples)        # Time to maturity between 0.01 and 2 years
r = np.random.uniform(0.01, 0.05, num_samples)     # Risk-free interest rate between 1% and 5%
sigma = np.random.uniform(0.1, 0.5, num_samples)   # Volatility between 10% and 50%

# Calculate option prices
call_prices = black_scholes_call_price(S, K, T, r, sigma)
put_prices = black_scholes_put_price(S, K, T, r, sigma)

# Calculate Greeks
delta = black_scholes_delta_call(S, K, T, r, sigma)
gamma = black_scholes_gamma(S, K, T, r, sigma)
theta = black_scholes_theta_call(S, K, T, r, sigma)

# Create a DataFrame
data = pd.DataFrame({
    'S': S,
    'K': K,
    'T': T,
    'r': r,
    'sigma': sigma,
    'call_price': call_prices,
    'delta': delta,
    'gamma': gamma,
    'theta': theta
})

# 2. Prepare the dataset
features = ['S', 'K', 'T', 'r', 'sigma', 'delta', 'gamma', 'theta']
X = data[features]
y = data['call_price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Build and train the neural network
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# 4. Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Calculate mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error on Test Set:", mse)
print("R-squared on Test Set:", r2)

# 5. Compare the model's predictions to the Black-Scholes model
# Since we generated data using Black-Scholes, we can check residuals
residuals = y_test - y_pred.flatten()

plt.figure(figsize=(10,6))
plt.hist(residuals, bins=50, edgecolor='k')
plt.title('Residuals Distribution')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of actual vs predicted prices
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Option Prices')
plt.xlabel('Actual Call Price')
plt.ylabel('Predicted Call Price')
plt.show()

# 6. Investigate market inefficiencies
# Introduce noise into the option prices to mimic market data
noise = np.random.normal(0, 0.5, num_samples)
call_prices_noisy = call_prices + noise

data['call_price_noisy'] = call_prices_noisy

# Update y to be the noisy prices
y_noisy = data['call_price_noisy']

# Split and scale data again
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X, y_noisy, test_size=0.2, random_state=42
)

X_train_scaled_n = scaler.fit_transform(X_train_n)
X_test_scaled_n = scaler.transform(X_test_n)

# Retrain the model on noisy data
model_noisy = Sequential()
model_noisy.add(Dense(64, input_dim=X_train_scaled_n.shape[1], activation='relu'))
model_noisy.add(Dense(64, activation='relu'))
model_noisy.add(Dense(1, activation='linear'))

model_noisy.compile(loss='mean_squared_error', optimizer='adam')

history_noisy = model_noisy.fit(
    X_train_scaled_n, y_train_n,
    validation_data=(X_test_scaled_n, y_test_n),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Predict on test data
y_pred_n = model_noisy.predict(X_test_scaled_n)

# Calculate mean squared error and R-squared
mse_n = mean_squared_error(y_test_n, y_pred_n)
r2_n = r2_score(y_test_n, y_pred_n)

print("Mean Squared Error on Test Set (Noisy Data):", mse_n)
print("R-squared on Test Set (Noisy Data):", r2_n)

# Plot residuals
residuals_n = y_test_n - y_pred_n.flatten()

plt.figure(figsize=(10,6))
plt.hist(residuals_n, bins=50, edgecolor='k')
plt.title('Residuals Distribution (Noisy Data)')
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of actual vs predicted prices
plt.figure(figsize=(10,6))
plt.scatter(y_test_n, y_pred_n, alpha=0.5)
plt.plot([y_test_n.min(), y_test_n.max()], [y_test_n.min(), y_test_n.max()], 'r--')
plt.title('Actual vs Predicted Option Prices (Noisy Data)')
plt.xlabel('Actual Call Price')
plt.ylabel('Predicted Call Price')
plt.show()
