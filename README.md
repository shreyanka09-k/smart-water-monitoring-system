# smart-water-monitoring-system
AI-based Smart Water Monitoring System using Temporal Random Forest to identify patterns associated with historical leak-proximity periods.

Project Overview

The Smart Water Monitoring System is an AI/ML prototype aligned with UN Sustainable Development Goal 6 (SDG 6): Clean Water and Sanitation.

The system analyzes hourly water-distribution-network sensor data and identifies patterns associated with historical leak-proximity periods. It uses temporal feature engineering and a Temporal Random Forest classifier to generate a risk score and classify observations as:

Normal
Potential Leak-Proximity

Predictions are presented through an interactive Streamlit dashboard.

Important: A Potential Leak-Proximity prediction does not confirm that a physical water leak exists. It indicates that the observed sensor pattern is associated with the historical leak-proximity class and should be followed by operational inspection.


SDG Alignment

SDG 6 — Clean Water and Sanitation (The project supports SDG 6 by exploring an AI-assisted approach to monitoring water-distribution-network data and identifying sensor patterns associated with historical leak-proximity periods. The intended contribution is better monitoring and prioritization of observations that may require further operational investigation.)

Problem Statement:- Water leakage in distribution networks can lead to significant water loss and inefficient use of resources. Large volumes of hourly sensor data can contain temporal patterns that are difficult to inspect manually.

This project explores whether machine learning can analyze these patterns and provide a risk indication associated with historical leak-proximity periods, supporting more efficient monitoring and prioritization of observations for further inspection.

Dataset:-The project uses a public hourly urban Water Distribution Network dataset.

Dataset: Hourly Anomaly Scores and Leak Labels from a Multi-Source Urban Water Distribution Network Dataset
DOI: 10.5281/zenodo.15096167
License: CC-BY 4.0

The dataset contains sensor/anomaly information from water-network and environmental sources. The binary fault_d7 label represents proximity (±7 days) to recorded leak events.

The dataset contains 8,737 hourly observations and 20 original columns. After temporal feature engineering, the project uses 59 columns, including the target/metadata and engineered features.
     
AI-SOLUTION
Water-network sensor data
          ↓
Data preprocessing
          ↓
Temporal feature engineering
          ↓
Chronological train / validation / test split
          ↓
Temporal Random Forest
          ↓
Risk score
          ↓
Normal / Potential Leak-Proximity
          ↓
Streamlit monitoring dashboard

Temporal Features:-Hour, day of week and month,Cyclical time encoding,1-hour, 6-hour and 24-hour lag features,6-hour and 24-hour rolling statistics,1-hour and 24-hour changes

These features allow the model to use recent and historical sensor behavior rather than relying only on individual observations.

Model:- Model: Temporal Random Forest

Key configuration:

300 trees

min_samples_leaf = 2

Balanced class weights

Median imputation

Random state = 42

Chronological data splitting

Data Split:-Split

Samples

Training

5,242

Validation

1,747

Test

1,748

The split is chronological to reduce the risk of using future observations to predict earlier observations.

Decision Threshold

The deployed application uses a decision threshold of 0.65.

The model output is treated as a risk score for the leak-proximity class, not as a confirmed physical leak probability.

Model Evaluation

The project evaluation produced approximately:

Metric

Result

Accuracy

85.53%

ROC-AUC

0.8426

PR-AUC

0.9636

Precision

0.88

Recall

0.96

F1 Score

0.92

These metrics describe performance on the held-out test portion of the source dataset and should not be interpreted as proof of performance on other water networks.

Streamlit Prototype

The application allows a user to:

Upload a water-sensor CSV file.

Automatically create the required temporal features.

Generate model risk scores.

Classify observations as Normal or Potential Leak-Proximity.

View the latest monitoring status.

Visualize the risk trend over time.

Inspect the highest-risk observations.

Download prediction results.

View model configuration information.

Run Locally

pip install -r requirements.txt
streamlit run app.py

Responsible AI

Fairness and Generalization

The model was developed using a specific public water-distribution dataset. Sensor behavior and network characteristics may differ across locations, so performance should be validated before deployment on another network.

Transparency

The project documents its model type, temporal features, decision threshold and evaluation metrics. The dashboard also provides model information to users.

Ethics

The system is intended as a decision-support and monitoring tool. A Potential Leak-Proximity prediction does not automatically establish that a physical leak exists.

Privacy

The project uses water-network sensor information and does not require personal user information for prediction.

Limitations

The target represents proximity to historical recorded leak events rather than direct physical leak confirmation.

The dataset represents a particular water-distribution context and may not generalize directly to other networks.

The current prototype uses uploaded/historical CSV data rather than a live IoT sensor stream.

The system does not measure actual water saved or financial savings.

Operational inspection is still required to confirm a physical leak.

Future Scope

Integration with live IoT/SCADA sensor streams

Validation using additional water-distribution networks

Model monitoring and drift detection

More detailed explainability for individual predictions

Geographic visualization of high-risk network locations where suitable sensor/location data are available

Integration with operational alerting systems

Repository Structure

smart-water-monitoring-system/
│
├── Smart_Water_Monitoring_Final.ipynb
├── app.py
├── temporal_random_forest.joblib
├── model_config.json
├── requirements.txt
└── README.md

Project Impact

The Smart Water Monitoring System demonstrates how temporal machine learning can convert large volumes of water-network sensor observations into an interpretable monitoring workflow.

The proposed system aims to support earlier identification of sensor patterns associated with leak-proximity periods, helping monitoring teams prioritize observations for further inspection and contributing to more efficient water-resource management.

Project Information

Project: Smart Water Monitoring System
Primary SDG: SDG 6 — Clean Water and Sanitation
Model: Temporal Random Forest
Interface: Streamlit
Dataset: Public urban Water Distribution Network dataset
Dataset DOI: 10.5281/zenodo.15096167
