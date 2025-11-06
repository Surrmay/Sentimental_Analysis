# =======================
# 7. Streamlit Dashboard
# =======================
# Save the following block to a separate file `app.py` to run with Streamlit

import streamlit as st
st.title("Product Review Sentiment Analysis")
user_input = st.text_area("Enter a product review")
if st.button("Predict Sentiment"):
    processed = preprocess(user_input)
    vect = vectorizer.transform([processed])
    prediction = model.predict(vect)[0]
    st.write(f"Predicted Sentiment: {prediction}")

#Optional: Visualize sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# =======================
# 8. Future Improvements
# =======================
# - Use deep learning (LSTM with Word2Vec or GloVe embeddings)
# - Add misclassification analysis
# - Improve preprocessing with contextual stopwords or POS tagging
# - Extend dashboard to include review trends by product category/date

# END
