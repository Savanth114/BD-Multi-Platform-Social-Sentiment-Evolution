# BD Multi-Platform Social Sentiment Evolution

## Overview

BD Multi-Platform Social Sentiment Evolution is a Big Data and Machine Learning project that analyzes sentiment from social media content across multiple platforms. The project uses Apache Spark for scalable data processing and Python-based machine learning techniques to classify sentiments and generate insights.

## Features

* Multi-platform social media sentiment analysis
* Large-scale data processing using Apache Spark
* Data cleaning and preprocessing
* Sentiment classification
* Platform-wise sentiment comparison
* Result generation and evaluation metrics
* Interactive web-based frontend

## Tech Stack

### Backend

* Python
* Flask
* Apache Spark

### Frontend

* HTML5
* CSS3
* JavaScript

### Data Processing & ML

* Pandas
* NumPy
* Scikit-learn

## Project Structure

BD-Multi-Platform-Social-Sentiment-Evolution/

├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── data/
│   ├── multi_platform_social_sentiment_evolution.csv
│   ├── column_descriptions_social_sentiment.md
│   └── README_social_sentiment.md
│
├── results/
│   ├── predictions_sample.csv
│   ├── platform_sentiment.csv
│   ├── label_distribution.csv
│   └── metrics.txt
│
├── src/
│   └── spark_multiplatform_sentiment.py
│
├── download_dataset.py
├── README.md
└── .gitignore

## Dataset

The dataset contains social media posts collected from multiple platforms with sentiment labels used for analysis and classification.

## Installation

1. Clone the repository

git clone https://github.com/Savanth114/BD-Multi-Platform-Social-Sentiment-Evolution.git

2. Navigate to the project directory

cd BD-Multi-Platform-Social-Sentiment-Evolution

3. Install dependencies

pip install -r backend/requirements.txt

## Running the Project

### Run Backend

python backend/app.py

### Run Sentiment Analysis

python src/spark_multiplatform_sentiment.py

### Open Frontend

Open:

frontend/index.html

in your browser.

## Results

The project generates:

* Sentiment predictions
* Platform-wise sentiment statistics
* Label distribution analysis
* Model evaluation metrics

Output files are stored in the `results/` directory.

## Future Enhancements

* Real-time social media sentiment tracking
* Advanced deep learning models
* Interactive analytics dashboard
* Multi-language sentiment analysis
* Cloud deployment support

## Author

Savanth G

B.E. Computer Science (AI & ML)

## License

This project is licensed under the MIT License.
