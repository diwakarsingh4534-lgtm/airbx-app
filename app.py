import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('airbx_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("AirbX — Arbitrage Persistence Predictor")
st.write("Predict whether a detected cross-exchange arbitrage opportunity will remain profitable long enough to trade.")

st.header("Enter Opportunity Details")

spread_pct = st.number_input("Spread %", min_value=0.0, max_value=100.0, value=2.0)
liquidity_score = st.number_input("Liquidity Score", min_value=0.0, value=100.0)
volatility = st.number_input("Volatility", min_value=0.0, value=0.05)
momentum = st.number_input("Momentum", value=0.0)
arb_amount = st.number_input("Arbitrage Amount", min_value=0.0, value=5.0)
arb_quantity = st.number_input("Arbitrage Quantity", min_value=0.0, value=0.1)

if st.button("Predict Persistence"):
    input_data = pd.DataFrame([[spread_pct, liquidity_score, volatility, momentum, arb_amount, arb_quantity]],
                                columns=['spread_pct', 'liquidity_score', 'volatility', 'momentum', 'arb_amount', 'arb_quantity'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"✅ This opportunity is likely PERSISTENT ({probability*100:.1f}% confidence)")
    else:
        st.error(f"❌ This opportunity is likely NOT persistent ({probability*100:.1f}% confidence)")
