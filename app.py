from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load the pre-trained model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Dictionary for emotion emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Initialize Flask app
app = Flask(__name__)

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        raw_text = request.form["raw_text"]
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
        confidence = np.max(probability)
        
        # Prepare data for Altair chart
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions", "probability"]
        
        # Create Altair chart
        chart = alt.Chart(proba_df_clean).mark_bar().encode(
            x='emotions',
            y='probability',
            color='emotions'
        ).properties(width=500, height=300)
        
        chart_html = chart.to_html()  # Convert chart to HTML for rendering

        emoji_icon = emotions_emoji_dict.get(prediction, "")
        
        return render_template("index.html", raw_text=raw_text, prediction=prediction,
                               emoji_icon=emoji_icon, confidence=confidence, chart_html=chart_html)
    
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    raw_text = data.get("text", "")
    
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)
    
    # Prepare response
    response = {
        "predicted_emotion": prediction,
        "probabilities": {
            emotion: prob for emotion, prob in zip(pipe_lr.classes_, probability[0])
        }
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
