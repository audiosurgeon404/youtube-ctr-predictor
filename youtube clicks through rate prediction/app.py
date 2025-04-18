from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('ctr_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        title = data['name']
        duration = float(data['duration'])
        stayed = float(data['stayed'])
        views = float(data['views'])
        impressions = float(data['impressions'])
        impression_seconds = float(data['impression_seconds'])
        category = data['category']

        # Category one-hot encoding
        categories = [
            'film fact', 'tech fact', 'actor fact', 'voiceactor fact',
            'character fact ', 'character fact', 'cartoon fact'
        ]

        # Base + engineered features
        features = {
            'Duration': duration,
            'Stayed To Watch': stayed,
            'Views': views,
            'Impressions': impressions,
            'Seconds': impression_seconds,

            'duration_per_impression': duration / (impressions + 1),
            'view_rate': views / (impressions + 1),
            'stay_rate': stayed / (duration + 0.1),
            'impression_speed': impressions / (impression_seconds + 1),
            'log_Views': np.log1p(views),
            'log_Impressions': np.log1p(impressions),
            'log_Seconds': np.log1p(impression_seconds),
            'title_length': len(title),
            'title_word_count': len(title.split()),
            'has_question': int('?' in title),
            'has_exclamation': int('!' in title),
            'has_number': int(any(char.isdigit() for char in title)),
            'uppercase_ratio': sum(1 for c in title if c.isupper()) / (len(title) + 1),
            'avg_ctr_per_duration': 0,
            'impressions_ratio': impressions / 50000,
            'views_ratio': views / 20000,
            'view_per_impression': views / (impressions + 1),
            'ctr_per_view': 0,
        }

        # Add one-hot encoded category flags
        for cat in categories:
            features[cat] = int(category == cat)

        # Create DataFrame
        input_df = pd.DataFrame([features])

        # Match trained model's feature order
        expected_columns = model.get_booster().feature_names
        input_df = input_df[expected_columns]

        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'Predicted CTR: {prediction:.2f}%')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)