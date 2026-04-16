from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import warnings
import os   # ✅ IMPORTANT FIX

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Backend is running 🚀"


def fetch_stock_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False)

        if df.empty:
            return None, None

        return df, {}
    except Exception as e:
        return None, None


def predict_stock(df, days_ahead=30):
    df = df.reset_index()
    df['Days'] = np.arange(len(df))

    X = df[['Days']].values
    y = df['Close'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    future_days = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
    future_scaled = scaler_X.transform(future_days)
    future_pred_scaled = model.predict(future_scaled)
    future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {
        "actual": y.tolist(),
        "predicted": y_pred.tolist(),
        "future": future_pred.tolist(),
        "mse": round(mse, 4),
        "r2": round(r2, 4),
        "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist()
    }


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        ticker = data.get('ticker', 'AAPL').upper()
        period = data.get('period', '1y')
        days_ahead = int(data.get('days_ahead', 30))

        df, info = fetch_stock_data(ticker, period)

        if df is None or len(df) < 30:
            return jsonify({'error': f'Invalid ticker: {ticker}'}), 400

        result = predict_stock(df, days_ahead)

        current_price = round(df['Close'].iloc[-1], 2)
        prev_price = round(df['Close'].iloc[-2], 2)

        change = round(current_price - prev_price, 2)
        change_pct = round((change / prev_price) * 100, 2)

        return jsonify({
            'ticker': ticker,
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,
            'high_52w': round(df['Close'].max(), 2),
            'low_52w': round(df['Close'].min(), 2),
            'predicted_next': round(result['future'][0], 2),
            'predicted_30d': round(result['future'][-1], 2),
            'mse': result['mse'],
            'r2': result['r2'],
            'chart': {
                'dates': result['dates'],
                'actual': result['actual'],
                'predicted': result['predicted'],
                'future': result['future']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
