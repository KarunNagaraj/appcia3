import streamlit as st
import nltk
import cv2
import numpy as np
import pandas as pd
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.set_page_config(
    page_title="News Visual Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.title("News Visual Intelligence")

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

st.subheader("01 — Input")
st.write("Article Source")

input_method = st.radio(
    "How would you like to provide the article?",
    ["Enter article URL", "Paste text", "Upload .txt file"],
    horizontal=True
)

text_content = ""

if input_method == "Paste text":
    text_content = st.text_area("Paste full article text here:", height=220)

elif input_method == "Upload .txt file":
    uploaded_text_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_text_file is not None:
        text_content = uploaded_text_file.read().decode("utf-8")

elif input_method == "Enter article URL":
    url_input = st.text_input("Paste the news article URL here:")
    if url_input:
        try:
            from newspaper import Article
            with st.spinner("Fetching article…"):
                article = Article(url_input)
                article.download()
                article.parse()
            if article.text:
                text_content = article.text
                st.success("Article extracted successfully!")
                if article.title:
                    st.write(f"**Title:** {article.title}")
                if article.authors:
                    st.write(f"**Authors:** {', '.join(article.authors)}")
                if article.publish_date:
                    st.write(f"**Published:** {article.publish_date.strftime('%B %d, %Y')}")
            else:
                st.warning(
                    "Could not extract article text from that URL. "
                    "The site may require JavaScript to render content. "
                    "Try pasting the text manually instead."
                )
        except ImportError:
            st.error("newspaper3k is not installed. Run: pip install newspaper3k")
        except Exception as e:
            st.error(f"Failed to fetch article: {e}")

if text_content:

    st.divider()

    st.subheader("02 — Preview")
    st.write("Article Extract")
    preview = text_content[:800] + ("…" if len(text_content) > 800 else "")
    st.info(preview)

    st.divider()

    sentiment = sia.polarity_scores(text_content)
    sentiment_score = sentiment["compound"]

    if sentiment_score > 0.05:
        sentiment_label = "Positive"
    elif sentiment_score < -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    tokens = word_tokenize(text_content.lower())
    filtered_words = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    freq_dist = nltk.FreqDist(filtered_words)
    common_words = freq_dist.most_common(10)

    categories = {
        "Politics": ["government", "election", "president", "minister", "parliament", "policy"],
        "Technology": ["technology", "ai", "software", "internet", "cyber", "data", "robot"],
        "Business": ["market", "stock", "company", "investment", "finance", "economy"],
        "Sports": ["match", "tournament", "goal", "player", "league", "score"],
        "Crime": ["police", "murder", "arrest", "crime", "investigation", "court"],
        "Health": ["hospital", "disease", "virus", "medical", "treatment", "health"],
        "Entertainment": ["movie", "film", "celebrity", "music", "actor", "show"]
    }
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for word in filtered_words if word in keywords)
        category_scores[category] = score
    predicted_category = max(category_scores, key=category_scores.get)

    st.subheader("03 — Analysis")
    st.write("Text Intelligence")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentiment Score", f"{sentiment_score:+.3f}", help="Compound VADER score")
    m2.metric("Sentiment", sentiment_label, help="Classification")
    m3.metric("Predicted Category", predicted_category, help="Rule-based NLP")
    m4.metric("Unique Words", len(set(filtered_words)), help="After stopword removal")

    col_freq, col_cat = st.columns(2, gap="large")

    with col_freq:
        st.write("Top 10 Keywords by Frequency")
        if common_words:
            df_words = pd.DataFrame(common_words, columns=["Word", "Frequency"])
            st.bar_chart(df_words.set_index("Word"))
        else:
            st.warning("Not enough valid words for frequency analysis.")

    with col_cat:
        st.write("Category Relevance Scores")
        df_categories = pd.DataFrame(
            list(category_scores.items()),
            columns=["Category", "Score"]
        )
        st.bar_chart(df_categories.set_index("Category"))

    st.divider()

    st.subheader("04 — Computer Vision")
    st.write("Face Detection")
    st.caption(
        "Upload any image to detect and count human faces using the OpenCV DNN ResNet-SSD model — "
        "a deep neural network capable of detecting faces regardless of angle, expression, skin tone, "
        "glasses, or whether eyes are open or closed. Each detection includes a confidence score."
    )

    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    PROTOTXT_PATH = os.path.join(BASE_DIR, "deploy.prototxt")
    MODEL_PATH    = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

    @st.cache_resource(show_spinner=False)
    def load_dnn_model(prototxt, caffemodel):
        return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
        st.warning(
            "Model files not found. Place deploy.prototxt and "
            "res10_300x300_ssd_iter_140000.caffemodel in the same folder as app.py."
        )
        net = None
    else:
        with st.spinner("Loading ResNet-SSD model…"):
            net = load_dnn_model(PROTOTXT_PATH, MODEL_PATH)

    uploaded_image = st.file_uploader(
        "Upload an image to detect faces (JPG / PNG)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image is not None and net is not None:

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        CONFIDENCE_THRESHOLD = 0.5
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        image_with_boxes = image_rgb.copy()
        face_count = 0
        confidences = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            face_count += 1
            confidences.append(confidence)

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (201, 168, 76), 2)

            label = f"{confidence * 100:.1f}%"
            label_y = y1 - 8 if y1 - 8 > 10 else y1 + 18
            cv2.putText(
                image_with_boxes, label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (201, 168, 76), 1, cv2.LINE_AA
            )

        avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0
        min_confidence = (min(confidences) * 100) if confidences else 0
        max_confidence = (max(confidences) * 100) if confidences else 0

        col_face, col_fstats = st.columns([3, 1], gap="large")

        with col_face:
            st.write("Annotated Output — Confidence % shown above each detected face")
            st.image(image_with_boxes, use_column_width=True)

        with col_fstats:
            st.metric("Faces Detected", face_count, help=f"Confidence threshold: {int(CONFIDENCE_THRESHOLD*100)}%")
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            st.metric("Confidence Range", f"{min_confidence:.0f}–{max_confidence:.0f}%")
            st.caption(f"ResNet-SSD · res10_300x300 · Caffe · Threshold {int(CONFIDENCE_THRESHOLD*100)}%")

else:
    st.info("Enter or upload article text to begin analysis.")