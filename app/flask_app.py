import os
import joblib
import pandas as pd
import plotly.express as px
from flask import Flask, request, jsonify, redirect
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Initialize Flask app
app = Flask(__name__)

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Load models and data
base_dir = os.path.join(os.path.expanduser("~"), 'Downloads', 'globalfishingwatch')
model = joblib.load(os.path.join(base_dir, 'models', 'anomaly_detection_model.pkl'))
scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
kmeans = joblib.load(os.path.join(base_dir, 'models', 'kmeans_model.pkl'))
clustered_data_path = os.path.join(base_dir, 'data', 'clustered_data.csv')
data = pd.read_csv(clustered_data_path)

# Define Dash layout
dash_app.layout = html.Div([
    html.H1("Vessel Monitoring Dashboard"),
    dcc.Dropdown(
        id='cluster-dropdown',
        options=[{'label': f'Cluster {i}', 'value': i} for i in range(10)],
        value=0
    ),
    dcc.Graph(id='cluster-graph'),
    html.H2("Prediction:"),
    html.Div([
        dcc.Input(id='lat_mean', type='number', placeholder='lat_mean', step=0.01),
        dcc.Input(id='lon_mean', type='number', placeholder='lon_mean', step=0.01),
        dcc.Input(id='event_start', type='text', placeholder='event_start (YYYY-MM-DDTHH:MM:SSZ)'),
        dcc.Input(id='event_end', type='text', placeholder='event_end (YYYY-MM-DDTHH:MM:SSZ)'),
        dcc.Input(id='event_type', type='text', placeholder='event_type'),
        html.Button('Submit', id='submit-val', n_clicks=0)
    ]),
    html.Div(id='prediction-output'),
    dcc.Graph(id='prediction-graph')
])

# Dash callbacks
@dash_app.callback(
    Output('cluster-graph', 'figure'),
    [Input('cluster-dropdown', 'value')]
)
def update_graph(cluster):
    cluster_data = data[data['cluster'] == cluster]
    fig = px.scatter_mapbox(
        cluster_data,
        lat='lat_mean',
        lon='lon_mean',
        color='duration',
        mapbox_style='open-street-map',
        title=f"Vessel Events in Cluster {cluster}"
    )
    return fig

@dash_app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-graph', 'figure')],
    [Input('submit-val', 'n_clicks')],
    [State('lat_mean', 'value'),
     State('lon_mean', 'value'),
     State('event_start', 'value'),
     State('event_end', 'value'),
     State('event_type', 'value')]
)
def update_prediction(n_clicks, lat_mean, lon_mean, event_start, event_end, event_type):
    if n_clicks > 0:
        try:
            input_data = [{
                'lat_mean': float(lat_mean),
                'lon_mean': float(lon_mean),
                'event_start': pd.to_datetime(event_start),
                'event_end': pd.to_datetime(event_end),
                'event_type': event_type
            }]
        except ValueError:
            return "Invalid input. Please check your inputs.", {}

        df = pd.DataFrame(input_data)
        df['duration'] = (df['event_end'] - df['event_start']).dt.total_seconds() / 3600.0
        features = ['lat_mean', 'lon_mean', 'duration', 'event_type']
        df = pd.get_dummies(df[features])
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        predictions = ["Anomaly" if p == 1 else "Normal" for p in predictions]

        fig = px.scatter_mapbox(
            df,
            lat='lat_mean',
            lon='lon_mean',
            color=predictions[0],
            mapbox_style='open-street-map',
            title="Prediction Result"
        )
        fig.update_layout(mapbox=dict(center=dict(lat=lat_mean, lon=lon_mean), zoom=10))

        return f"Prediction: {predictions[0]}", fig

    return "", {}

# Flask routes
@app.route('/')
def home():
    return redirect('/dashboard/')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data)
        df['event_start'] = pd.to_datetime(df['event_start'])
        df['event_end'] = pd.to_datetime(df['event_end'])
        df['duration'] = (df['event_end'] - df['event_start']).dt.total_seconds() / 3600.0
        features = ['lat_mean', 'lon_mean', 'duration', 'event_type']
        df = pd.get_dummies(df[features])
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        predictions = ["Anomaly" if p == 1 else "Normal" for p in predictions]
        return jsonify(predictions)
    except Exception as e:
        return jsonify(str(e)), 400

if __name__ == '__main__':
    app.run(debug=True)
