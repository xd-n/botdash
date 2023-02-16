import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

import numpy as np
import lightgbm as lgb


# Load data
sales_data = pd.read_csv("sales_data_4.csv")
inventory_data = pd.read_csv("inventory_data_2.csv", parse_dates=["date"], index_col="date")
uv_data = pd.read_csv("uv_data.csv", parse_dates=["date"], index_col="date")
uv_data = uv_data.reset_index()
inventory_data = inventory_data.reset_index()
#sales_data = sales_data.reset_index()
# Set up the app
app = dash.Dash(__name__)
server = app.server


# Train LightGBM model on sales data
X_train = sales_data.index.values.reshape(-1, 1)
y_train = sales_data['Sales'].values
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Generate sales forecast for the next 7 days using the trained LightGBM model
next_week = pd.date_range(start='2022-03-01', end='2022-03-07', freq='D')
X_test = np.array([i for i in range(len(sales_data), len(sales_data) + 7)]).reshape(-1, 1)
y_pred = model.predict(X_test)
sales_forecast = pd.DataFrame({
    'Date': next_week,
    'Sales': y_pred
})

# Create visualization of sales data and sales forecast
fig_sales = {
    'data': [
        {'x': sales_data['Date'], 'y': sales_data['Sales'], 'type': 'line', 'name': 'Sales'},
        {'x': sales_forecast['Date'], 'y': sales_forecast['Sales'], 'type': 'line', 'name': 'Sales Forecast'}
    ],
    'layout': {
        'title': 'Sales and Sales Forecast',
        'xaxis': {'title': 'Date'},
        'yaxis': {'title': 'Sales'}
    }
}


app.layout = html.Div([
    html.H1('Dashboard'),

    html.Div([
        dcc.Graph(id='uv-graph', figure=px.line(uv_data, x='date', y='uv', title='Unique Viewers')),
        dcc.Graph(id='inventory-graph', figure=px.line(inventory_data, x='date', y='inventory', title='Inventory'))
    ], className='row'),

    html.Div([
        dcc.Graph(id='sales-graph', figure=fig_sales),
        #dcc.Graph(id='sales-graph', figure=px.line(sales_data, x='date', y='sales', title='Sales')),
        #dcc.Graph(id='forecast-graph', figure=px.line(sales_forecast_df, x='date', y='sales', title='Sales Forecast'))
    ], className='row')
    
])


# Define callbacks
@app.callback(
    Output(component_id="uv-graph", component_property="figure"),
    [Input(component_id="interval-component", component_property="n_intervals")]
)
def update_uv_graph(n):
    fig = px.line(uv_data, x=uv_data.index, y="uv")
    fig.update_layout(title="UV Tracking")
    return fig

@app.callback(
    Output(component_id="inventory-graph", component_property="figure"),
    [Input(component_id="interval-component", component_property="n_intervals")]
)
def update_inventory_graph(n):
    fig = px.line(inventory_data, x=inventory_data.index, y="inventory")
    fig.update_layout(title="Inventory Tracking")
    return fig

@app.callback(
    Output(component_id="sales-graph", component_property="figure"),
    [Input(component_id="interval-component", component_property="n_intervals")]
)
def update_sales_graph(n):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sales_data.index, y=sales_data["sales"], mode="lines", name="Actual"))
    fig.update_layout(title="Sales Tracking")
    return fig


if __name__ == "__main__":
    app.run(debug=True)
    #changed by myself
    #app.run_server(debug=True, port=8023, use_reloader=False)
