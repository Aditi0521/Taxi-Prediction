# 🚕 Manhattan Taxi Demand Prediction 
A sophisticated forecasting system for predicting taxi demand across Manhattan using a hybrid approach combining machine learning and deep neural networks. The system uses a CNN-LSTM Encoder-Decoder model for Manhattan's top 5 locations and a Stacked Machine Learning model for remaining locations.

## 📑 Table of Contents
* [Overview](#overview)
* [Features](#features)
* [Data Sources](#data-sources)
* [Architecture](#architecture)
* [Models](#models)
* [Usage](#usage)
* [Results](#results)
* [Technologies Used](#technologies-used)

## 🎯 Overview
This project develops a comprehensive taxi demand forecasting system for Manhattan, leveraging historical trip data from 2020-2022. The system combines advanced deep learning techniques with traditional machine learning approaches to provide accurate predictions for different areas of Manhattan.

## ⭐ Features
- 🔄 Real-time demand prediction visualization
- 🗺️ Interactive heatmap of Manhattan zones
- 📊 Hourly and daily demand pattern analysis
- ⛈️ Weather impact analysis 
- 🎊 Holiday and weekend demand patterns
- 📍 Top locations analysis
- 📱 Detailed analytics dashboard

## 📚 Data Sources
- 🚖 Taxi trip records from [NYC TLC](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (2020-2022)
- 🌤️ Weather data from [Visual Crossing](https://www.visualcrossing.com/)
- 🗽 NYC taxi zone information
- 🎉 US holiday data

## 🏗️ Architecture
The system uses a hybrid architecture:
- 🧠 CNN-LSTM Encoder-Decoder: For top 5 Manhattan locations
- 📚 Stacked Machine Learning Model: For remaining locations 
- ☁️ AWS EMR Spark Cluster: For data processing
- 💻 Streamlit: For web application interface

## 🤖 Models
### CNN-LSTM Encoder-Decoder
- 📈 Performance Metrics:
  - R-Square: 95.6%
  - Mean Square Error: 839.1
  - Mean Absolute Error: 19.56
  - Root Mean Square Error: 28.96

### Stacked Model
- 📊 Performance Metrics:
  - R-Square: 96.72%
  - Mean Square Error: 125.51
  - Mean Absolute Error: 5.82
  - Root Mean Square Error: 11.2

## 🎮 Usage
- 📊 Dashboard View
  - View real-time demand predictions
  - Analyze demand patterns through interactive maps
  - Monitor key metrics and trends
- 📈 Detailed Analysis
  - Explore time series analysis
  - Compare demand across locations
  - Analyze weather impact

## 🎯 Results
- ✅ Successfully predicted taxi demand with over 96% accuracy
- 🔍 Identified key patterns in demand based on:
  - ⏰ Time of day
  - 🌦️ Weather conditions
  - 🎉 Special events
  - 📍 Location characteristics

## 🛠️ Technologies Used
  - 🐍 Python
  - 🧠 TensorFlow
  - 📊 Scikit-learn
  - 🌐 Streamlit
  - ⚡ PySpark
  - ☁️ Google Cloud Platform
