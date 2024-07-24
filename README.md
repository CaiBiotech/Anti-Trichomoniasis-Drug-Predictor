# Anti-Trichomoniasis Drug Prediction Tool

This repository contains a machine learning-based tool for predicting the activity of compounds against Trichomoniasis. The tool uses molecular fingerprints and various machine learning algorithms to make predictions.

## Features

- Supports multiple molecular fingerprint types (FP2, FP3, FP4, MACCS, ECFP2, ECFP4, ECFP6, FCFP2, FCFP4, FCFP6)
- Implements various machine learning algorithms (Decision Trees, K-Nearest Neighbors, Linear Discriminant Analysis, Gradient Boosting, Naive Bayes, Logistic Regression, Random Forest, AdaBoost, Multi-Layer Perceptron, XGBoost, Support Vector Machines, Extra Trees)
- Provides both a desktop application (using PyQt5) and a web interface (using Streamlit)
- Includes comprehensive benchmarking of different algorithms and fingerprints
- Performs hyperparameter tuning for the best-performing model
- Generates visualizations for model performance and prediction distributions

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/anti-trichomoniasis-prediction.git

2. Install the required dependencies:
pip install -r requirements.txt

## Usage
### Desktop Application

Run the desktop application:
1）Extract (or unzip) the compressed file named "Anti-Trichomoniasis Drug Predictor.tgz"
2）Once the extraction is complete, locate and run the executable file named "Anti-Trichomoniasis Drug Predictor.exe"

### Web Interface

Run the Streamlit web interface:
streamlit run cem_pgdrug_web.py

## Model Training

The model training process includes the following steps:
1. Data preparation and fingerprint calculation
2. Comparison of different algorithms and fingerprint types
3. Selection of the best-performing algorithm and fingerprint
4. Hyperparameter tuning for the best model
5. Final model evaluation and saving

To retrain the model or reproduce the results, run:
python fp2model_fix4.py

## Files

- `Anti-Trichomoniasis Drug Predictor.exe`: PyQt5-based desktop application
- `cem_pgdrug_web.py`: Streamlit-based web interface
- `fp2model_fix4.py`: Script for model training and evaluation
- `best_model.joblib`: Saved best-performing model
- `icon.png`: Application icon
- `requirements.txt`: List of required Python packages

## Results

The training process generates several output files:
- `benchmark_results_1.csv`: Results of algorithm and fingerprint comparisons
- `benchmark_results_2.csv`: Detailed metrics for the best fingerprint
- `benchmark_results_3.csv`: Hyperparameter tuning results
- `prediction_results.csv`: Predictions on the entire dataset
- `prediction_stats.csv`: Statistics of predictions across molecular weight ranges
- Various plots (ROC curve, benchmark heatmaps, prediction distribution)

## License

<GPL-3.0 license>

## Contact

Dr. Haiming Cai
E-mail: haiming_cai@hotmail.com

Project Link: https://github.com/CaiBiotech/Anti-Trichomoniasis-Drug-Predictor
