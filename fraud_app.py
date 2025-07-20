# fraud_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("Smart Freeze: AI Fraud Detection App")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Feature Engineering
    df['Amount/Balance'] = df['TransactionAmount'] / df['AccountBalance']
    df['High Login Attempts'] = df['LoginAttempts'].apply(lambda x: 1 if x > 3 else 0)
    df['Time Gap'] = df['Time Gap'].apply(lambda x: -500 if x == -30 else x)

    # Define features and target
    features = ['TransactionAmount', 'Time Gap', 'Amount/Balance', 'High Login Attempts']
    if 'is_fraud' in df.columns:
        target = 'is_fraud'
        X = df[features]
        y = df[target]

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict
        df['predicted_fraud'] = model.predict(X)

        # Smart Freeze Alert
        def freeze_alert(row):
            if row['predicted_fraud'] == 1:
                return ['Transfer Money', 'Change Password']
            else:
                return []

        df['Frozen Features'] = df.apply(freeze_alert, axis=1)

        # Display results
        st.success("Prediction Completed!")
        st.dataframe(df[['TransactionID', 'predicted_fraud', 'Frozen Features']].head(20))

        # Alerts
        for i, row in df.iterrows():
            if row['predicted_fraud'] == 1:
                st.error(f"ALERT: Fraudulent transaction detected for Transaction ID {row['TransactionID']}! Features Frozen: {', '.join(row['Frozen Features'])}")

        # Option to download result
        result_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", data=result_csv, file_name="fraud_detection_results.csv", mime="text/csv")

    else:
        st.warning("The uploaded CSV must contain an 'is_fraud' column.")
