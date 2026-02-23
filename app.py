import streamlit as st
import nltk
import cv2
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ----------------------------
# INITIAL SETUP
# ----------------------------

st.set_page_config(page_title="News Visual Intelligence", layout="wide")
st.title("📰 Text + Image Intelligence Dashboard")

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# ----------------------------
# MANUAL TEXT INPUT SECTION
# ----------------------------

st.subheader("📝 Enter Article Text")

text_content = st.text_area(
    "Paste full article text here:",
    height=250
)

uploaded_text_file = st.file_uploader(
    "Or upload a .txt file",
    type=["txt"]
)

if uploaded_text_file is not None:
    text_content = uploaded_text_file.read().decode("utf-8")

# Only proceed if text exists
if text_content:

    st.subheader("📄 Article Preview")
    st.write(text_content[:1000])  # Show first 1000 characters

    # ----------------------------
    # NLP ANALYSIS
    # ----------------------------

    st.subheader("📊 Text Analysis")

    sentiment = sia.polarity_scores(text_content)
    sentiment_score = sentiment["compound"]

    if sentiment_score > 0.05:
        sentiment_label = "Positive"
    elif sentiment_score < -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    st.write("Sentiment Score:", sentiment_score)
    st.write("Sentiment Label:", sentiment_label)

    # Word Frequency Analysis
    tokens = word_tokenize(text_content.lower())
    filtered_words = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    freq_dist = nltk.FreqDist(filtered_words)
    common_words = freq_dist.most_common(10)

    if common_words:
        df_words = pd.DataFrame(common_words, columns=["Word", "Frequency"])
        st.bar_chart(df_words.set_index("Word"))
    else:
        st.info("Not enough valid words for frequency analysis.")

        # ----------------------------
    # RULE-BASED NEWS CLASSIFICATION
    # ----------------------------

    st.subheader("🗂 News Category Classification")

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

    st.write("Predicted Category:", predicted_category)

    # Optional: show scores
    df_categories = pd.DataFrame(
        list(category_scores.items()),
        columns=["Category", "Score"]
    )

    st.bar_chart(df_categories.set_index("Category"))

    # ----------------------------
    # IMAGE UPLOAD
    # ----------------------------

    st.subheader("🖼 Upload Related Image")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image is not None:

        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image_rgb, caption="Original Image", use_column_width=True)

        # ----------------------------
        # BASIC IMAGE METRICS
        # ----------------------------

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])

        edges = cv2.Canny(image, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        st.subheader("🔎 Edge Detection")
        st.image(edges, caption="Canny Edge Detection", use_column_width=True)

        st.write("Brightness:", round(brightness, 2))
        st.write("Edge Density:", round(edge_density, 4))

        # ----------------------------
        # FACE + EYE DETECTION (Filtered)
        # ----------------------------

        st.subheader("👁 Face & Eye Detection")

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(60, 60)
        )

        image_with_boxes = image_rgb.copy()
        valid_face_count = 0

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image_with_boxes[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            # Only accept face if eyes detected
            if len(eyes) >= 1:
                valid_face_count += 1

                cv2.rectangle(
                    image_with_boxes,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    2
                )

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (0, 127, 255),
                        2
                    )

        st.image(
            image_with_boxes,
            caption="Validated Faces with Eyes",
            use_column_width=True
        )

        st.write("Valid Faces Detected:", valid_face_count)

        # ----------------------------
        # VISUAL SENTIMENT SCORING
        # ----------------------------

        visual_score = 0

        # Brightness effect
        if brightness < 80:
            visual_score -= 0.5
        elif brightness > 150:
            visual_score += 0.5

        # Chaos via edge density
        if edge_density > 0.2:
            visual_score -= 0.3

        # Face logic
        if valid_face_count >= 3:
            visual_score -= 0.3
        elif valid_face_count == 1:
            visual_score += 0.2

        st.subheader("🧠 Text–Image Consistency")

        consistency = 1 - abs(sentiment_score - visual_score)
        consistency = max(0, round(consistency, 3))

        st.write("Visual Sentiment Score:", round(visual_score, 3))
        st.write("Consistency Score:", consistency)

        if consistency > 0.7:
            st.success("Text and Image are emotionally aligned.")
        else:
            st.warning("Possible emotional mismatch between text and image.")

else:
    st.info("Enter or upload article text to begin analysis.")