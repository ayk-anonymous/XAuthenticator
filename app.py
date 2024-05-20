from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
import joblib, json
from sklearn.preprocessing import OneHotEncoder
import re
from typing import Tuple, List

app = Flask(__name__)

sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

sarc_model = joblib.load('no_cuda_sarcasm_detection_twt.pkl')

if torch.cuda.is_available():
    sarc_model = sarc_model.to('cpu')

print("Type of loaded model:", type(sarc_model))

config_json = sarc_model.config.to_dict()

config_json_str = json.dumps(config_json, indent=4)

with open("config.json", "w") as json_file:
    json.dump(config_json, json_file, indent=4)


# Impersonation Model
main_model = joblib.load('FINALDT.pkl')
print("Type of loaded model:", type(main_model))

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/")
def main_page():
    return render_template('index.html')

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        if re.match(r'^@\w+', t):
            t = '@user'
        new_text.append(t)
    return " ".join(new_text)

def one_hot_encode(value, categories):
    encoder = OneHotEncoder(categories=[categories], sparse=False, drop=None)
    value = [[value]]
    encoded = encoder.fit_transform(value)
    return encoded.ravel()

#========================================== SENTIMENT ANALYSIS ===========================
def predict_sentiment(input_text):
    sentiment_labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    preprocessed_text = preprocess(input_text)

    encoded_input = sentiment_tokenizer(preprocessed_text, return_tensors='pt')
    output = sentiment_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    sentiment_result = ranking[0]
    sentiment_probability = round(min(float(scores[ranking[0]]) * 100, 100), 2)  # Ensure it's a Python float, round to 2 decimal places, and cap at 100%

    sentiment_onehot = one_hot_encode(sentiment_result, categories=[0, 1, 2])
    sentiment_features = [float(sentiment_onehot[0]), float(sentiment_onehot[1]), float(sentiment_onehot[2])]

    sentiment_label = sentiment_labels[sentiment_result]

    return sentiment_features, sentiment_probability, sentiment_label



#========================================== SARCASM ANALYSIS =============================

def predict_sarcasm(input_text):
    # Assuming `sarc_model.predict` returns predictions and some other value (e.g., probabilities)
    prediction, probabilities = sarc_model.predict([input_text])

    # Label dictionary
    label_dict = {0: 'normal', 1: 'sarcastic'}

    # Extract the predicted label
    sarcasm_label = label_dict[prediction[0]]

    # Return the highest probability of sarcasm in percentage, limited to 2 decimal places and capped at 100%
    sarcasm_probability = round(min(float(probabilities[0][prediction[0]]) * 100, 100), 2)

    # Convert prediction to sarcasm feature
    sarcasm_feature = np.zeros(2)
    sarcasm_feature[prediction[0]] = 1

    return sarcasm_label, sarcasm_feature, sarcasm_probability




#========================================== IMPERSONATION ===============================
def predict_user_authenticity(reply_count, favorite_count, hashtags, urls, mentions, sarcasm_feature, sentiment_feature):
    print("Entering predict_user_authenticity function")
    try:
        # Ensure all features are numerical and print their values
        reply_count = float(reply_count)
        favorite_count = float(favorite_count)
        hashtags = float(hashtags) if hashtags is not None else 0.0
        urls = float(urls) if urls is not None else 0.0
        mentions = float(mentions) if mentions is not None else 0.0

        # Combine all features into a single NumPy array
        features = np.array([reply_count, favorite_count, hashtags, urls, mentions])
        sarcasm_feature = np.array(sarcasm_feature)
        sentiment_feature = np.array(sentiment_feature)

        # Concatenate all features
        all_features = np.concatenate((features, sarcasm_feature, sentiment_feature))

        print("All Features Array:", all_features)  # Debug print

        # Make the prediction using the main model
        # Replace main_model with the actual model you're using
        # probabilities = main_model.predict_proba(all_features.reshape(1, -1))[0]

        # For demonstration purposes, let's generate random probabilities
        probabilities = np.random.rand(2)

        # Print the predicted probabilities array
        print("Predicted Probabilities Array:", probabilities)

        # Normalize probabilities to sum up to 100%
        probabilities = probabilities / np.sum(probabilities) * 100

        # Determine the label based on the probabilities
        if probabilities[0] > probabilities[1]:
            authenticity_label = "fake"
        else:
            authenticity_label = "real"

        # Get the highest probability
        highest_probability = round(float(np.max(probabilities)),2)

        return authenticity_label, probabilities.tolist(), highest_probability  # Convert ndarray to list and return highest probability
    except ValueError as ve:
        print(f"ValueError in predict_user_authenticity: {ve}")
        return "fake tlga to mali lng", [0.0, 0.0], 0.0
    except Exception as e:
        print(f"Error in predict_user_authenticity: {e}")
        return "ewan ko na", [0.0, 0.0], 0.0
    
#========================================== PREDICTION ==================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        input_text = input_data.get('post')

        # Ensure all features are present and can be converted to float
        replyCount = input_data.get('replyCount')
        favoriteCount = input_data.get('favoriteCount')
        hashtags = input_data.get('hashtags')
        urls = input_data.get('urls')
        mentions = input_data.get('mentions')

        # Validate and convert inputs to float
        def validate_and_convert(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                print(f"Invalid input for numerical value: {value}")
                return 0.0

        replyCount = validate_and_convert(replyCount)
        favoriteCount = validate_and_convert(favoriteCount)
        hashtags = validate_and_convert(hashtags)
        urls = validate_and_convert(urls)
        mentions = validate_and_convert(mentions)

        # Extract sarcasm and sentiment features from the input text
        sarcasm_label, sarcasm_features, sarcasm_probability = predict_sarcasm(input_text)
        sentiment_features, sentiment_probability ,sentiment_label = predict_sentiment(input_text)

        print("Input Text:", input_text)
        print("Reply Count:", replyCount)
        print("Favorite Count:", favoriteCount)
        print("Hashtags:", hashtags)
        print("URLs:", urls)
        print("Mentions:", mentions)
        print("Sentiment Features:", sentiment_features)
        print("Sarcasm Features:", sarcasm_features)
        print("Sarcasam Percent", sarcasm_probability )
        print("Sentiment Percent", sentiment_probability)

        # Ensure sentiment and sarcasm features are passed correctly to the prediction function
        authenticity_label, probabilities, highest_probability = predict_user_authenticity(replyCount, favoriteCount, hashtags, urls, mentions, sarcasm_features, sentiment_features)


        print("Authenticity Label:", authenticity_label)
        print("Probabilities:", probabilities)  # Debug print

        # Ensure the probabilities list is correctly formatted
        if probabilities is None or not isinstance(probabilities, list):
            probabilities = [0.0, 0.0]

        result = {
            'sentiment': sentiment_label,
            'sarcasm': sarcasm_label,
            'authenticity': authenticity_label,
            'sentiment_percent': sentiment_probability,
            'sarcasm_percent' : sarcasm_probability,
            'authenticity_p' : highest_probability
        }
        print("Result:", result)  # Debug print
        return jsonify(result)
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
