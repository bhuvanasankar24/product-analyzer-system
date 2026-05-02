import streamlit as st
import pandas as pd
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="InsightLens", layout="wide")

st.title("📊 InsightLens — Product Review Analyzer")
st.markdown("Upload a CSV file with product reviews to analyze sentiment and complaint topics dynamically.")

# -----------------------
# Helper Functions
# -----------------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)  # remove HTML for modeling
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\b\w{1,2}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ✅ NEW: clean only for display (fixes <br>)
def clean_display_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)   # remove HTML tags like <br>
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_sentiment(score):
    if score >= 0.2:
        return "positive"
    elif score <= -0.2:
        return "negative"
    else:
        return "neutral"

# -----------------------
# Upload
# -----------------------
uploaded_file = st.file_uploader("📂 Upload your review CSV", type=["csv"])

if uploaded_file is not None:

    try:
        df = load_data(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # -----------------------
        # Column Handling
        # -----------------------
        if 'review' not in df.columns:
            if 'Text' in df.columns:
                df = df.rename(columns={'Text': 'review'})
            else:
                st.error("❌ CSV must contain 'review' or 'Text' column.")
                st.stop()

        df = df[['review']].dropna().reset_index(drop=True)
        st.write(f"📌 Total Reviews: {len(df)}")

        # -----------------------
        # Cleaning
        # -----------------------
        df['clean_review'] = df['review'].apply(clean_text)

        df = df[df['clean_review'].str.len() > 20]

        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            st.info("⚡ Sampled 10,000 reviews for faster processing")

        # -----------------------
        # Sentiment Analysis
        # -----------------------
        sia = SentimentIntensityAnalyzer()
        df['compound'] = df['clean_review'].apply(lambda x: sia.polarity_scores(x)['compound'])
        df['sentiment'] = df['compound'].apply(get_sentiment)

        # -----------------------
        # Sentiment Visualization
        # -----------------------
        st.subheader("📈 Sentiment Distribution")
        st.bar_chart(df['sentiment'].value_counts())

        # -----------------------
        # Topic Modeling
        # -----------------------
        neg_df = df[df['sentiment'] == 'negative'].copy()

        if len(neg_df) > 10:

            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(neg_df['clean_review'])

            nmf = NMF(n_components=5, random_state=42)
            W = nmf.fit_transform(X)
            H = nmf.components_

            terms = vectorizer.get_feature_names_out()

            neg_df['topic'] = np.argmax(W, axis=1)

            # -----------------------
            # Extract Keywords
            # -----------------------
            topic_keywords = {}

            for i, topic in enumerate(H):
                top_words = [terms[j] for j in topic.argsort()[-5:]]
                topic_keywords[i] = top_words

            stop_words_extra = {"dont", "just", "like", "bad", "really", "good", "buy", "order", "item"}

            topic_keywords_clean = {}

            for i, words in topic_keywords.items():
                filtered = [w for w in words if w not in stop_words_extra]
                topic_keywords_clean[i] = filtered[:3] if filtered else words[:3]

            topic_names = {
                i: " / ".join(words) + " issues"
                for i, words in topic_keywords_clean.items()
            }

            neg_df['topic_name'] = neg_df['topic'].map(topic_names)

            # -----------------------
            # Topic Visualization
            # -----------------------
            st.subheader("🚨 Top Complaint Topics")
            st.bar_chart(neg_df['topic_name'].value_counts())

            # -----------------------
            # Show Topics
            # -----------------------
            st.subheader("🔍 Discovered Topics")

            for i, words in topic_keywords_clean.items():
                st.write(f"**Topic {i}:** {', '.join(words)}")

            # -----------------------
            # Insights
            # -----------------------
            st.subheader("🧠 Key Insights")

            topic_percent = (neg_df['topic_name'].value_counts(normalize=True) * 100).round(2)

            for topic, percent in topic_percent.items():
                st.write(f"- {percent}% of complaints are related to **{topic}**")

            # -----------------------
            # Sample Reviews (FIXED)
            # -----------------------
            st.subheader("💬 Sample Complaints")

            top_topics = neg_df['topic_name'].value_counts().index

            for topic_name in top_topics:
                st.write(f"### {topic_name}")
                samples = neg_df[neg_df['topic_name'] == topic_name]['review'].head(3)

                if len(samples) > 0:
                    for i, s in enumerate(samples):
                        preview = clean_display_text(s[:200])

                        with st.expander(f"Sample {i+1}: {preview}..."):
                            st.write(clean_display_text(s))
                else:
                    st.write("No sample available.")

        else:
            st.warning("⚠️ Not enough negative reviews to perform topic analysis.")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("⬆️ Please upload a CSV file to begin analysis.")