import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Streamlit settings
st.set_page_config(layout="wide")

# Sidebar for user input
st.sidebar.header('Monte Carlo Simulation Settings')
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1000, max_value=100000, value=10000, step=1000)
num_data_points = st.sidebar.number_input('Number of Data Points', min_value=100, max_value=1000, value=256, step=50)

# File uploader for closing prices CSV
closing_prices_file = st.sidebar.file_uploader("Upload Closing Prices CSV", type=["csv"])
portfolio_file = st.sidebar.file_uploader("Upload Portfolio CSV", type=["csv"])

if closing_prices_file is not None and portfolio_file is not None:
    # Read the closing prices CSV into a DataFrame
    closing_prices_df = pd.read_csv(closing_prices_file)

    # Calculate daily returns
    closing_prices_df.set_index('Date', inplace=True)
    daily_returns_df = closing_prices_df.pct_change().dropna()

    # Ensure daily_returns_df has exactly the specified number of data points
    if len(daily_returns_df) > num_data_points:
        daily_returns_df = daily_returns_df.tail(num_data_points)
    elif len(daily_returns_df) < num_data_points:
        st.error(f"The daily returns DataFrame has fewer than {num_data_points} data points.")
        st.stop()

    # Read the portfolio CSV into a DataFrame
    portfolio_df = pd.read_csv(portfolio_file)

    # Calculate portfolio weights
    total_value = portfolio_df['Market Value'].sum()
    portfolio_df['Weight'] = portfolio_df['Market Value'] / total_value

    # Add Average and Standard Deviation columns based on daily returns
    average_returns = daily_returns_df.mean()
    std_dev_returns = daily_returns_df.std()

    portfolio_df['Average'] = portfolio_df['Stock'].map(average_returns)
    portfolio_df['Standard Deviation'] = portfolio_df['Stock'].map(std_dev_returns)

    # Initialize the simulation DataFrame
    simulation_df = pd.DataFrame(index=range(num_simulations), columns=portfolio_df['Stock'])

    # Run Monte Carlo simulations
    for stock in portfolio_df['Stock']:
        mean = portfolio_df.loc[portfolio_df['Stock'] == stock, 'Average'].values[0]
        std_dev = portfolio_df.loc[portfolio_df['Stock'] == stock, 'Standard Deviation'].values[0]
        weight = portfolio_df.loc[portfolio_df['Stock'] == stock, 'Weight'].values[0]
        
        # Generate random Z-scores for the stock
        z_scores = norm.ppf(np.random.rand(num_simulations), mean, std_dev)
        
        # Store the Z-scores in the simulation DataFrame
        simulation_df[stock] = z_scores * weight

    # Calculate the Portfolio Return by summing the weighted Z-scores
    simulation_df['Portfolio Return'] = simulation_df.sum(axis=1)

    # Calculate the Value at Risk (VaR) at different confidence levels
    VaR_levels = [0.90, 0.95, 0.99]
    portfolio_VaRs = {level: np.percentile(simulation_df['Portfolio Return'], 100 * (1 - level)) for level in VaR_levels}

    # Display DataFrames to ensure they are read correctly and new columns are added
    st.write("## Closing Prices DataFrame")
    st.write(closing_prices_df)

    st.write("## Daily Returns DataFrame")
    st.write(daily_returns_df)

    st.write("## Portfolio DataFrame")
    st.write(portfolio_df)

    st.write("## Monte Carlo Simulation DataFrame")
    st.write(simulation_df)

    # Display VaR analysis
    st.write("## Value at Risk (VaR) Analysis at Different Confidence Levels")
    for level, var in portfolio_VaRs.items():
        st.write(f"At the {int(level*100)}% confidence level, the Value at Risk (VaR) is {var:.4f}. This means that there is a {int(level*100)}% chance that the portfolio will not lose more than {var:.4f} in a single day. Conversely, there is a {100 - int(level*100)}% chance that the portfolio will lose more than {var:.4f} in a single day.")

    # Plotting the histogram of Portfolio Returns with VaR levels
   
