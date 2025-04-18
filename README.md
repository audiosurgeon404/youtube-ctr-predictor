
# YouTube CTR Predictor
This project predicts the Click-Through Rate (CTR) of YouTube videos using a machine learning model based on metadata such as title, duration, impressions, views, and more. It includes a clean web interface styled like YouTube and is built with Flask.
# History
I decided to make this project to put the knowledge into practice and most importantly to help my youtuber friend to pay attention to what criteria most affect the success of his videos, as well as his ctr . I managed to create an accurate model with 0.82 score, it was not easy, however I managed :) 
I also created for a friend a simple and easy to understand site for predicting ctr.

## Demo

![screenshot](https://github.com/audiosurgeon404/youtube-ctr-predictor/demo.png)

## Features

- Predicts CTR using features like title length, impressions, view ratios, etc.
- Handles category-based one-hot encoding for video types (e.g. Film Fact, Tech Fact)
- HTML form for user input with styling inspired by YouTube
- Engineered features: log values, ratios, text analysis
- Outlier filtering and feature standardization during training

## Tech Stack

- Python, Flask
- XGBoost model
- Pandas & NumPy for preprocessing
- HTML/CSS frontend

## How to Use

1. Clone the repository
```bash
git clone https://github.com/audiosurgeon/youtube-ctr-predictor.git
cd youtube-ctr-predictor
