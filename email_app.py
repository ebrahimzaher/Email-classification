import streamlit as st
import pickle

# تحميل النموذج والـ vectorizer
with open('svm_spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    tfidf = pickle.load(vec_file)

st.title("📩 Spam Message Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # تحويل النص إلى تمثيل TF-IDF
        transformed_text = tfidf.transform([user_input])
        prediction = model.predict(transformed_text)[0]
        
        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (Not Spam).")
