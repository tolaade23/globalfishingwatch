from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = Flask(__name__)

# Load the trained models and scaler
model = joblib.load('models/anomaly_detection_model.pkl')
scaler = joblib.load('models/scaler.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    df['event_start'] = pd.to_datetime(df['event_start'])
    df['event_end'] = pd.to_datetime(df['event_end'])
    df['duration'] = (df['event_end'] - df['event_start']).dt.total_seconds() / 3600.0
    features = ['lat_mean', 'lon_mean', 'duration', 'event_type']
    df = pd.get_dummies(df[features])
    X_scaled = scaler.transform(df)
    predictions = model.predict(X_scaled)
    predictions = [1 if p == -1 else 0 for p in predictions]
    return jsonify(predictions)

# Dash app for visualization
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Layout of the dashboard
dash_app.layout = html.Div([
    html.H1("Vessel Monitoring Dashboard"),
    dcc.Dropdown(
        id='cluster-dropdown',
        options=[{'label': f'Cluster {i}', 'value': i} for i in range(10)],
        value=0
    ),
    dcc.Graph(id='cluster-graph')
])

# Callback for updating graph based on selected cluster
@dash_app.callback(
    Output('cluster-graph', 'figure'),
    [Input('cluster-dropdown', 'value')]
)
def update_graph(cluster):
    cluster_data = pd.read_csv('data/CVP_combined_data.csv')  # Assuming combined data is saved here
    cluster_data = cluster_data[cluster_data['cluster'] == cluster]
    fig = px.scatter_mapbox(
        cluster_data,
        lat='lat_mean',
        lon='lon_mean',
        color='duration',
        mapbox_style='open-street-map',
        title=f"Vessel Events in Cluster {cluster}"
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
