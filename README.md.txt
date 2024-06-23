# Global Fishing Watch

This project involves analyzing and monitoring global fishing activities using various machine learning techniques and data visualization.

## Directory Structure

- `data/`: Contains the CSV files of the datasets.
- `models/`: Contains the saved models and scaler.
- `notebooks/`: Contains the Jupyter notebooks for data processing.
- `scripts/`: Contains the Python scripts for preprocessing, anomaly detection, clustering, and other analyses.
- `app/`: Contains the Flask and Dash applications.
- `README.md`: Project documentation.
- `requirements.txt`: Python dependencies.

## Steps

1. **Data Preprocessing**: Combining and preprocessing datasets.
2. **Anomaly Detection**: Detecting anomalies in vessel data using Isolation Forest.
3. **Clustering**: Clustering vessel events using K-Means.
4. **Advanced Machine Learning Projects**:
    - LSTM Model for Vessel Movement Prediction.
    - Temporal Pattern Recognition.
    - Geospatial Analysis and Visualization.
5. **Web Application**: Flask and Dash applications for predictions and visualization.

## Running the Project

1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocessing: `python scripts/preprocess.py`
3. Run anomaly detection: `python scripts/anomaly_detection.py`
4. Run clustering: `python scripts/clustering.py`
5. Run the Flask application: `python app/flask_app.py`

## Dependencies

See `requirements.txt` for the list of dependencies.


## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/tolaade23/vessel-monitoring-dashboard.git
   cd vessel-monitoring-dashboard
