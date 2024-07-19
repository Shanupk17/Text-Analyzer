from flask import Flask, request, render_template
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os
import re

app = Flask(__name__)

# Utility function for preprocessing text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# Utility function to save the model and tokenizer
def save_model_and_tokenizer(lstm_model, gru_model, tokenizer):
    lstm_model.save('lstm_model.h5')
    gru_model.save('gru_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

# Train models based on the input paragraph
def train_models(texts, labels):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=100)
    y = np.array(labels)

    # Define and train LSTM model
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    lstm_model.add(LSTM(128, return_sequences=True))
    lstm_model.add(LSTM(64))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X, y, epochs=1, batch_size=1)  # Use batch_size=1 for simplicity

    # Define and train GRU model
    gru_model = Sequential()
    gru_model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    gru_model.add(GRU(128, return_sequences=True))
    gru_model.add(GRU(64))
    gru_model.add(Dense(1, activation='sigmoid'))
    gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gru_model.fit(X, y, epochs=1, batch_size=1)  # Use batch_size=1 for simplicity

    save_model_and_tokenizer(lstm_model, gru_model, tokenizer)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    paragraph = request.form['paragraph']
    word = request.form['word']
    
    processed_paragraph = preprocess_text(paragraph)
    
    # Prepare data for training
    texts = [processed_paragraph]
    labels = [0]  # Dummy label for simplicity; you might need actual labels based on context
    
    # Train models and save them
    train_models(texts, labels)
    
    # Load models
    lstm_model = load_model('lstm_model.h5')
    gru_model = load_model('gru_model.h5')

    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Prepare input for prediction
    sequences = tokenizer.texts_to_sequences([processed_paragraph])
    X = pad_sequences(sequences, maxlen=100)
    
    # Predict with models
    lstm_prediction = lstm_model.predict(X)
    gru_prediction = gru_model.predict(X)

    # Compute word frequency
    words = processed_paragraph.split()
    word_count = words.count(word.lower())
    
    # Check if the word is unique
    is_unique = word_count == 1

    result = {
        'frequency': word_count,
        'is_unique': is_unique,
        'lstm_prediction': lstm_prediction[0][0],  # Adjust based on your model's output
        'gru_prediction': gru_prediction[0][0]   # Adjust based on your model's output
    }
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    if not os.path.isfile('lstm_model.h5') or not os.path.isfile('gru_model.h5'):
        print("No pre-trained models found. Please train the models first.")
    app.run(debug=True)
