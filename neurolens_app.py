import streamlit as st
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import requests
from io import BytesIO

# Set up the Streamlit page
st.title("NeuroLens: AI Vision & Chat")
st.write("Upload an image for a caption, ask a question, or chat with NeuroLens!")

# Load models with caching
@st.cache_resource
def load_models():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return captioner, vqa, tokenizer, model

captioner, vqa, gpt_tokenizer, gpt_model = load_models()

# Function to generate chat response
def generate_chat_response(prompt, max_length=50):
    inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = gpt_model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=gpt_tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        do_sample=True
    )
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip() or "Let's chat more!"

# Image input section
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
            st.error("Invalid URL or image.")

# Display image and vision features
if image:
    st.image(image, caption="Selected Image", use_column_width=True)
    st.subheader("Image Caption")
    caption = captioner(image)[0]["generated_text"]
    st.write(f"**NeuroLens**: {caption}")
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question (e.g., 'Is the cat black?')")
    if question:
        result = vqa(image, question, top_k=1)
        answer = result[0]["answer"]
        st.write(f"**NeuroLens**: {answer} ({result[0]['score']*100:.1f}% sure)")

# Spacer to push chat to bottom
st.markdown("<div style='height: 100vh;'></div>", unsafe_allow_html=True)

# Chat section at bottom
st.subheader("Chat with NeuroLens")
with st.container():
    user_input = st.text_input("Talk to NeuroLens (e.g., 'Tell me about cats!')", key="chat_input")
    if user_input:
        chat_prompt = f"As a friendly AI, respond to: {user_input}"
        chat_response = generate_chat_response(chat_prompt, max_length=50)
        st.write(f"**NeuroLens**: {chat_response}")
