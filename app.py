import datetime as dt
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


st.set_page_config(
	page_title="Stock Market Prediction Platform",
	page_icon="📈",
	layout="wide",
)


@st.cache_data(show_spinner=False)
def fetch_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
	"""Download historical OHLCV data for the selected ticker."""
	df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
	if df.empty:
		return df
	df.index = pd.to_datetime(df.index)
	return df


@st.cache_data(show_spinner=False)
def fetch_usd_inr_rate() -> float:
	"""Get the latest USD to INR FX rate using Yahoo Finance; fall back to a conservative default."""
	fx = yf.download("USDINR=X", period="5d", progress=False)
	if fx.empty:
		return 83.0
	return float(fx["Close"].iloc[-1])


def build_features(df: pd.DataFrame, lag_days: int, ma_window: int) -> Tuple[pd.DataFrame, pd.Series]:
	"""Create lagged price features and technical signals for next-day close prediction."""
	data = df.copy()
	data = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
	data = data.fillna(method="ffill").fillna(method="bfill")

	for i in range(1, lag_days + 1):
		data[f"lag_close_{i}"] = data["Close"].shift(i)

	data["ma_close"] = data["Close"].rolling(ma_window).mean()
	data["volatility"] = data["Close"].pct_change().rolling(ma_window).std()
	data["target"] = data["Close"].shift(-1)

	data = data.dropna()
	features = data.drop(columns=["target"])
	target = data["target"]
	return features, target


def train_test_split_time(
	X: pd.DataFrame, y: pd.Series, train_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Time-ordered split to avoid leakage."""
	split_idx = int(len(X) * train_ratio)
	X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
	y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
	return X_train, X_test, y_train, y_test


def evaluate_model(
	model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
	preds = model.predict(X_test)
	mse = mean_squared_error(y_test, preds)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, preds)
	return {"MSE": mse, "RMSE": rmse, "R2": r2, "preds": preds}


def plot_predictions(dates: pd.Index, actual: pd.Series, predicted: np.ndarray) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.plot(dates, actual, label="Actual", color="#2563eb", linewidth=2)
	ax.plot(dates, predicted, label="Predicted", color="#f97316", linewidth=2, linestyle="--")
	ax.set_title("Actual vs Predicted Close (Test Set)", fontsize=12, pad=12)
	ax.set_ylabel("Price")
	ax.legend()
	ax.grid(alpha=0.2)
	fig.autofmt_xdate()
	return fig


def predict_next_day(model: LinearRegression, latest_features: pd.DataFrame) -> float:
	return float(model.predict(latest_features.tail(1))[0])


def header_section():
	st.title("Stock Market Prediction Platform")
	st.caption(
		"Fast, transparent linear regression forecasts with clean visuals powered by Yahoo Finance data."
	)


def sidebar_inputs(default_start: dt.date, default_end: dt.date) -> Dict[str, object]:
	st.sidebar.header("Configure")
	companies = {
		"Apple (AAPL)": "AAPL",
		"Microsoft (MSFT)": "MSFT",
		"Alphabet (GOOGL)": "GOOGL",
		"Amazon (AMZN)": "AMZN",
		"Meta (META)": "META",
		"NVIDIA (NVDA)": "NVDA",
		"Tesla (TSLA)": "TSLA",
		"JPMorgan Chase (JPM)": "JPM",
		"Visa (V)": "V",
		"Berkshire Hathaway (BRK-B)": "BRK-B",
		"Custom": "Custom",
	}
	selection = st.sidebar.selectbox("Ticker", options=list(companies.keys()), index=0)
	if companies[selection] == "Custom":
		ticker = st.sidebar.text_input("Enter custom ticker", value="AAPL").upper().strip()
	else:
		ticker = companies[selection]
	start = st.sidebar.date_input("Start date", value=default_start)
	end = st.sidebar.date_input("End date", value=default_end)
	train_ratio = st.sidebar.slider("Train split", 0.6, 0.95, 0.8, 0.05)
	lag_days = st.sidebar.slider("Lag days", 2, 15, 5)
	ma_window = st.sidebar.slider("Moving average window", 3, 30, 10)
	run = st.sidebar.button("Run prediction", type="primary")
	return {
		"ticker": ticker,
		"start": start,
		"end": end,
		"train_ratio": train_ratio,
		"lag_days": lag_days,
		"ma_window": ma_window,
		"run": run,
	}


def stats_cards(latest_close: float, predicted_next: float):
	delta = predicted_next - latest_close
	pct_delta = (delta / latest_close) * 100 if latest_close else 0.0
	col1, col2, col3 = st.columns(3)
	col1.metric("Last close", f"₹{latest_close:,.2f}")
	col2.metric("Predicted next close", f"₹{predicted_next:,.2f}", f"{delta:+.2f}")
	col3.metric("Predicted move", f"{pct_delta:+.2f}%")


def main():
	header_section()
	today = dt.date.today()
	defaults = sidebar_inputs(default_start=today - dt.timedelta(days=365 * 3), default_end=today)

	if not defaults["run"]:
		st.info("Set your parameters in the sidebar and click 'Run prediction'.")
		return

	with st.spinner("Fetching data..."):
		raw = fetch_data(defaults["ticker"], defaults["start"], defaults["end"])
		fx_rate = fetch_usd_inr_rate()

	if raw.empty or len(raw) < defaults["lag_days"] + 15:
		st.error("Not enough data returned. Try a wider date range or different ticker.")
		return

	features, target = build_features(raw, defaults["lag_days"], defaults["ma_window"])
	if len(features) < 30:
		st.error("Insufficient feature rows after preprocessing. Loosen parameters.")
		return

	X_train, X_test, y_train, y_test = train_test_split_time(features, target, defaults["train_ratio"])
	model = LinearRegression()
	model.fit(X_train, y_train)

	eval_res = evaluate_model(model, X_test, y_test)
	mse_inr = eval_res["MSE"] * (fx_rate ** 2)
	rmse_inr = eval_res["RMSE"] * fx_rate
	predicted_next_usd = predict_next_day(model, features)
	latest_close_usd = float(raw["Close"].iloc[-1])
	predicted_next = predicted_next_usd * fx_rate
	latest_close = latest_close_usd * fx_rate

	stats_cards(latest_close, predicted_next)

	st.subheader("Model performance")
	perf_cols = st.columns(3)
	perf_cols[0].metric("MSE (INR)", f"{mse_inr:.2f}")
	perf_cols[1].metric("RMSE (INR)", f"{rmse_inr:.2f}")
	perf_cols[2].metric("R²", f"{eval_res['R2']:.3f}")

	st.subheader("Visuals")
	fig = plot_predictions(
		dates=X_test.index,
		actual=y_test * fx_rate,
		predicted=eval_res["preds"] * fx_rate,
	)
	st.pyplot(fig, use_container_width=True)

	st.subheader("Recent data preview")
	preview = raw.copy()
	preview["Close_INR"] = preview["Close"] * fx_rate
	st.dataframe(preview.tail(10))

	st.caption(
		"Prices, projections, and error metrics shown in INR using the latest USD/INR rate. The platform keeps everything in-memory and uses linear regression to maintain speed and transparency."
	)

	business_health_dashboard(defaults["ticker"], fx_rate)

def business_health_dashboard(ticker: str, fx_rate: float):
	st.markdown("---")
	st.header("📊 5-Year Business Health Dashboard")

	stock = yf.Ticker(ticker)
	hist_5y = stock.history(period="5y")

	if hist_5y.empty:
		st.warning("Unable to load 5-year business data.")
		return

	# ---- KPI Calculations ----
	first_price = hist_5y["Close"].iloc[0]
	last_price = hist_5y["Close"].iloc[-1]
	growth_5y = ((last_price - first_price) / first_price) * 100

	one_year = stock.history(period="1y")
	one_year_growth = (
		(one_year["Close"].iloc[-1] - one_year["Close"].iloc[0])
		/ one_year["Close"].iloc[0]
	) * 100

	financials = stock.financials

	if not financials.empty and "Total Revenue" in financials.index:
		revenue = financials.loc["Total Revenue"][0]
		net_income = financials.loc["Net Income"][0]
		profit_margin = (net_income / revenue) * 100
	else:
		revenue = 0
		net_income = 0
		profit_margin = 0

	revenue_inr = revenue * fx_rate

	# ---- KPI Cards ----
	col1, col2, col3, col4 = st.columns(4)
	col1.metric("5-Year Growth %", f"{growth_5y:.2f}%")
	col2.metric("1-Year Growth %", f"{one_year_growth:.2f}%")
	col3.metric("Latest Revenue (INR)", f"₹{revenue_inr:,.0f}")
	col4.metric("Profit Margin %", f"{profit_margin:.2f}%")

	# ---- 5-Year Price Chart ----
	st.subheader("📈 5-Year Stock Trend")

	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(hist_5y.index, hist_5y["Close"] * fx_rate)
	ax.set_title("Stock Price (INR) - Last 5 Years")
	ax.set_ylabel("Price (INR)")
	ax.grid(alpha=0.3)
	st.pyplot(fig, use_container_width=True)

	# ---- Revenue & Net Income ----
	if not financials.empty:
		st.subheader("💰 Revenue vs Net Income")

		revenue_series = financials.loc["Total Revenue"] * fx_rate
		income_series = financials.loc["Net Income"] * fx_rate

		finance_df = pd.DataFrame({
			"Revenue (INR)": revenue_series,
			"Net Income (INR)": income_series,
		})

		st.bar_chart(finance_df.T)

	# ---- Business Health Indicator ----
	st.subheader("🏥 Business Health Status")

	if profit_margin > 20 and one_year_growth > 10:
		st.success("🟢 Business is Strong & Growing")
	elif profit_margin > 10:
		st.warning("🟡 Business is Stable")
	else:
		st.error("🔴 Business Growth is Weak")



if __name__ == "__main__":
	main()