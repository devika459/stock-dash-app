from alpha_vantage.timeseries import TimeSeries

import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import svr_model

import plotly.graph_objs as go
import plotly.express as px

from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

API_KEY = "8XSFYGIJTJ2CVDQC"
ts = TimeSeries(key=API_KEY, output_format="pandas")


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
html.Div([
html.P("Welcome to the Stock Dash App!", className="start"),

html.Div([
    html.Label("Input stock code:"),
    dcc.Input(
        id="stock-code",
        type="text",
        placeholder="Enter stock code",
        value=""
    ),
    html.Button("Submit", id="submit", n_clicks=0),
]),

html.Div([
    dcc.DatePickerRange(
        id="date-range",
        start_date=dt(2021, 1, 1),
        end_date=dt.today().date()
    ),
]),

html.Div([
    html.Button("Stock Price", id="stock-price-button", n_clicks=0),
    html.Button("Indicators", id="indicators-button", n_clicks=0),
    dcc.Input(
        id="forecast-days",
        type="number",
        placeholder="number of days",
        min=1,
        max=365
    ),
    html.Button("Forecast", id="forecast", n_clicks=0),
]), 
], className="inputs"),

html.Div([
html.Div([
    # Logo
    # Company Name
], className="header"),

html.Div(id="description", className="description_ticker"),

html.Div([], id="graphs-content"),

html.Div([], id="main-content"),

html.Div([], id="forecast-content"),
], className="content")


], className="container")

def get_stock_price_fig(df):
    fig = px.line(
        df,
        x="Date",
        y=["Open", "Close"],
        title="Closing and Opening Price vs Date"
    )
    return fig

def get_more(df):
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    fig = px.scatter(
        df,
        x="Date",
        y="EMA_20",
        title="Exponential Moving Average vs Date"
    )
    fig.update_traces(mode="lines+markers")
    return fig

    
import time

company_cache = {}

@app.callback(
    Output("description", "children"),
    Input("submit", "n_clicks"),
    State("stock-code", "value")
)


def update_company_info(n, stock):

    if not n:
        return ""

    if not stock:
        return "Enter stock code"

    stock = stock.upper().strip()

    return html.Div([
        html.H3(stock),
        html.P(
            "Company information is currently unavailable. "
            "This demo focuses on stock price and indicator visualizations."
        )
    ])

    
@app.callback(
    Output("graphs-content", "children"),
    Input("stock-price-button", "n_clicks"),
    State("stock-code", "value")
)
def update_stock_price_graph(n, stock):

    if not n:
        return html.P("Click Stock Price button")

    if not stock:
        return html.P("Enter stock code")

    try:
        data, meta = ts.get_daily(symbol=stock, outputsize="compact")

        data.reset_index(inplace=True)
        data.rename(columns={
            "date": "Date",
            "1. open": "Open",
            "4. close": "Close"
        }, inplace=True)

        fig = px.line(
            data,
            x="Date",
            y=["Open", "Close"],
            title=f"{stock.upper()} Open vs Close"
        )

        return dcc.Graph(figure=fig)

    except Exception as e:
        return html.P(f"Alpha Vantage error: {e}")



@app.callback(
    Output("main-content", "children"),
    Input("indicators-button", "n_clicks"),
    State("stock-code", "value")
)
def update_indicator_graph(n, stock):

    if not n:
        return html.P("Click Indicators button")

    if not stock:
        return html.P("Enter stock code")

    try:
        data, meta = ts.get_daily(symbol=stock, outputsize="compact")
        data.reset_index(inplace=True)

        data["EMA_20"] = data["4. close"].ewm(span=20).mean()

        fig = px.line(
            data,
            x="date",
            y=["4. close", "EMA_20"],
            title=f"{stock.upper()} EMA Indicator"
        )

        return dcc.Graph(figure=fig)

    except Exception as e:
        return html.P(f"Alpha Vantage error: {e}")


@app.callback(
    
    Output("forecast-content", "children"),
    Input("forecast", "n_clicks"),
    State("stock-code", "value"),
    State("forecast-days", "value")
)
def forecast_stock(n, stock, days):

    if not n:
        return html.P("Click Forecast button to generate forecast")

    if not stock:
        return html.P("Enter stock code first")

    if not days:
        days = 7

    stock = stock.upper().strip()

    try:
        # ✅ Fetch Alpha Vantage data
        data, meta = ts.get_daily(symbol=stock, outputsize="compact")

        if data is None or data.empty:
            return html.P("No data received from Alpha Vantage.")

        # ✅ Convert into our format
        data.reset_index(inplace=True)

        data.rename(columns={
            "date": "Date",
            "1. open": "Open",
            "4. close": "Close"
        }, inplace=True)

        df = data[["Date", "Open", "Close"]].copy()

        model, mse, mae = svr_model.build_svr_model(df)
        future_pred = svr_model.forecast_next_days(model, df, days=days)


        last_date = pd.to_datetime(df["Date"].max())
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": future_pred
        })

        fig = px.line(
            forecast_df,
            x="Date",
            y="Forecast",
            title=f"{stock} Forecast (SVR Model)"
        )

        return html.Div([
            html.P(f"MSE: {mse:.4f} | MAE: {mae:.4f}"),
            dcc.Graph(figure=fig)
        ])

    except Exception as e:
        return html.P(f"Forecast error: {e}")



if __name__ == '__main__':
    app.run(debug=True)