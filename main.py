import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# =====================================================================
# 1. KERAS 3 / LEGACY H5 COMPATIBILITY PATCH
# =====================================================================
class PatchedSimpleRNN(tf.keras.layers.SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Strip the problematic argument
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)  # Strip from saved metadata
        return super().from_config(config)

# =====================================================================
# 2. CACHED MODEL & DICTIONARY LOADING
# =====================================================================
@st.cache_resource # Keeps the model in memory so it loads instantly after the first time
def load_sentiment_model():
    # Use the custom object scope to safely load your legacy file
    with tf.keras.utils.custom_object_scope({'SimpleRNN': PatchedSimpleRNN}):
        return load_model('simple_rnn_imdb.h5')

@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

# Initialize the model and indexes
model = load_sentiment_model()
word_index, reverse_word_index = load_word_index()

# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    # IMDB dataset offset index mapping (+3)
    # Restrict to words under the 10000 vocabulary cap if needed, 
    # but clipping happens natively at the embedding layer
    encoded_review = []
    for word in words:
        raw_idx = word_index.get(word, None)
        
        if raw_idx is None:
            # Word doesn't exist anywhere in the IMDB dictionary -> OOV (2)
            encoded_review.append(2)
        else:
            # Shift by 3 to align with Keras training dataset rules
            actual_idx = raw_idx + 3
            
            # If the index exceeds your 10,000 vocabulary size -> OOV (2)
            if actual_idx >= 10000:
                encoded_review.append(2)
            else:
                encoded_review.append(actual_idx)
                
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# =====================================================================
# 4. STREAMLIT USER INTERFACE
# =====================================================================
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review', placeholder="Type your review here...")

if st.button('Classify'):
    if user_input.strip() == "":
        st.warning("Please enter some text before classifying!")
    else:
        # Preprocess and predict
        preprocess_input = preprocess_text(user_input)
        prediction = model.predict(preprocess_input)
        score = prediction[0][0]
        
        # Determine sentiment classification
        sentiment = 'Positive' if score > 0.5 else 'Negative'

        # Visual layout for results
        st.subheader("Analysis Results")
        if sentiment == 'Positive':
            st.success(f'**Sentiment:** {sentiment} ')
        else:
            st.error(f'**Sentiment:** {sentiment} ')
            
        st.info(f'**Prediction Score:** {score:.4f}')
else:
    st.write('Please enter a movie review above.')