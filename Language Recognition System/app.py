import streamlit as st
import pickle
import re

import os
st.write("Current working directory:", os.getcwd())
st.write("Available files:", os.listdir())

# Loading the model
model=pickle.load(open("Language Detector Model.pkl","rb"))

# Loading the bag of words
bow=pickle.load(open("Bag of words.pkl","rb"))

# Loading the encoder
encoder=pickle.load(open("LabelEncoder.pkl","rb"))

# TITLE
st.title("Language Detector App")

# getting sentence from user
input_text=st.text_input("Please enter the text")

# cleaning the obtained text
input_text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', input_text)
input_text = re.sub(r'[[]]', ' ',input_text)
input_text = input_text.lower()

# Applying bag of words on the obtained text
bow_vectors=bow.transform([input_text])
feature_names=bow.get_feature_names_out()

     
if input_text:

    # Checking whether it is doing correct bag of words of unseen data
    st.write("The words that model understood are")
    for each in bow_vectors[0].nonzero()[1]:
        st.write(feature_names[each])


    # giving the bow_vectors to model to predict
    predicted_language_number=model.predict(bow_vectors)


    # Reverse transformation of obtained language number
    predicted_language=encoder.inverse_transform(predicted_language_number)
    st.write("The input text is of ",predicted_language[0],"language")

