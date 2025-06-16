import streamlit as st
import pandas as pd
import fitz   # to extract text from pdf
import re
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Loading pre-trained bert model and other nlp material 
vectorizer=TfidfVectorizer()
lemmatizer=WordNetLemmatizer()

#nlp=spacy.load("en_core_web_sm")

# Admin Dictionary
admin={
    "najafalialkhani@gmail.com":"031213",
    "zoroaster786@gmail.com":"icanBE01"
}

# Adding title
st.title("NLP-Screen: Intelligent Resume Parsing and Matching")


# Initialize session state
if 'admin_login' not in st.session_state:
    st.session_state.admin_login = False

# Sidebar title
st.sidebar.title("Admin Panel")

# Login button logic
if not st.session_state.admin_login:
    if st.sidebar.button("Login as Admin"):
        st.session_state.show_login_form = True

# Show login form after clicking the button
if st.session_state.get('show_login_form', False) and not st.session_state.admin_login:
    email = st.sidebar.text_input("Enter Email", key="email")
    password = st.sidebar.text_input("Enter Password", type="password", key="password")

    if email and password:
        if email in admin and admin[email] == password:
            st.sidebar.success("Login Successful")
            st.session_state.admin_login = True
        else:
            st.sidebar.error("Incorrect credentials")

# After successful login, show job description input
if st.session_state.admin_login:
    st.sidebar.markdown("### Enter Job Description")
    job_desc = st.sidebar.text_area("Job Description", height=200)

    # Cleaning job description
    job_desc=re.sub("[^a-zA-Z]"," ",job_desc)
    job_desc=job_desc.lower()
    job_desc=job_desc.split()
    job_desc=[lemmatizer.lemmatize(word) for word in job_desc if word not in stopwords.words("english")]

    # job_description in vectors
    if job_desc:
        st.session_state.job_desc_text = " ".join(job_desc)  # store cleaned job desc as string
        st.session_state.job_desc_vectors = vectorizer.fit_transform([st.session_state.job_desc_text])
    else:
        st.session_state.job_desc_text = " ".join("Looking for a Python developer with experience in machine learning.")  # store cleaned job desc as string
        st.session_state.job_desc_vectors = vectorizer.fit_transform([st.session_state.job_desc_text])
    
    if job_desc:
        st.sidebar.success("Job description saved. You can now match resumes.")

    
        # Store job_desc in a variable or session_state
        st.session_state.job_description = job_desc

# Getting CV from user
cv_uploaded=st.file_uploader("Upload CV",type=['.pdf','.txt','.docx'])


# Finding the format of uploaded cv
if cv_uploaded:
    file_type=cv_uploaded.type

    # If the cv is uploaded in text format
    if file_type=="text/plain":
        text_cv=cv_uploaded.read().decode("utf-8")
        text_cv=re.sub("[^a-zA-Z0-9]"," ",text_cv)
        text_cv=text_cv.lower()
        text_cv=text_cv.split()
        text_cv=[lemmatizer.lemmatize(word) for word in text_cv if word not in stopwords.words("english")]
        

    # If the cv is uploaded in PDF format
    elif file_type=="application/pdf":
        with fitz.open(stream=cv_uploaded.read(), filetype="pdf") as doc:
            cv = ""
            for page in doc:
                cv += page.get_text()

        # cleaning the cv
        text_cv=re.sub("[^a-zA-Z0-9]"," ",cv)
        text_cv=text_cv.lower()
        text_cv=text_cv.split()
        text_cv=[lemmatizer.lemmatize(word) for word in text_cv if word not in stopwords.words("english")]
        

    # If the cv is uploaded in Document format    
    else:
        # Load the DOCX file
        cv_uploaded.seek(0)  # Ensure the file pointer is at the beginning
        doc = Document(cv_uploaded)

        # Extract text from all paragraphs
        word_cv= "\n".join([para.text for para in doc.paragraphs])

        # Cleaning the extracted data from cv
        word_cv=re.sub("[^a-zA-Z0-9]"," ",word_cv)
        word_cv=word_cv.lower()
        word_cv=word_cv.split()
        text_cv=[lemmatizer.lemmatize(word) for word in word_cv if word not in stopwords.words("english")]


    if 'job_desc_vectors' in st.session_state:
            cv_text = " ".join(text_cv)
            pdf_cv_vectors = vectorizer.transform([cv_text])
            similarity_score = cosine_similarity(pdf_cv_vectors, st.session_state.job_desc_vectors)
            if similarity_score>=0.65:
                st.write("Please review this resume, it matches your requirements")
            else:
                st.write("The Resume is to be ignored, as it doesnot match the requirements")
    else:
        st.warning("Please enter a job description first in the sidebar.")
        
        


    