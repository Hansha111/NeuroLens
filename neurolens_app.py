import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Set up the Streamlit page
st.title("NeuroLens: AI Vision & Language")
st.write("Upload an image or enter a URL, see its description, and ask a question!")

# Load models
@st.cache_resource
def load_models():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
    return captioner, vqa

captioner, vqa = load_models()

# Image input
option = st.radio("Choose input method:", ("Upload Image", "Enter Image URL"))
image = None

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
else:
    url = st.text_input("Enter image URL (e.g., https://example.com/cat.jpg)")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("Invalid URL or image. Please try another.")

if image:
    # Display the image
    st.image(image, caption="Selected Image", use_column_width=True)

    # Image Captioning
    st.subheader("Image Caption")
    caption = captioner(image)[0]["generated_text"]
    st.write(f"**NeuroLens says**: {caption}")

    # Visual Question Answering
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question (e.g., 'Is the cat black?')")
    if question:
        result = vqa(image, question, top_k=1)
        answer = result[0]["answer"]
        confidence = result[0]["score"]
        st.write(f"**NeuroLens answers**: {answer} ({confidence*100:.1f}% sure)")