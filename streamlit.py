#Import libraries and Load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

#Load the word index of IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#Load the pretrained model
model = load_model('simple_rnn_imdb.h5')

#Now we need helper functions,
#One to decode the reviews, one to  preprocess the user input

#Function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    # Clip values to fit within the embedding layer's vocabulary range
    encoded_review = np.clip(encoded_review, 0, 9999)
    #Using padding
    padded_review = sequence.pad_sequences([encoded_review], maxlen = 500)
    return padded_review


##Streamlit app
st.title('Movie Review Sentiment Analyzer')
st.write('Enter a movie review to classify it as Positive or Negative.')

#User input
user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    #Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    
    #Display the result
    st.write(f"The reviewer has a {sentiment} sentiment.")
    st.write(f"Prediction score is {prediction[0][0]}")
    
else:
    st.write("Please enter a movie review.")
