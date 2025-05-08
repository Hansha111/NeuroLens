import streamlit as st
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import requests
from io import BytesIO

# Set up the Streamlit page
st.title("NeuroLens: AI Vision & Language")
st.write("Upload an image or enter a URL, see its description, ask a question, or chat with NeuroLens!")

# Load models with caching
@st.cache_resource
def load_vision_models():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
    return captioner, vqa

@st.cache_resource
def load_gpt_model():
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return tokenizer, model

captioner, vqa = load_vision_models()
gpt_tokenizer, gpt_model = load_gpt_model()

# Function to generate dialogue
def generate_dialogue(prompt, max_length=50):
    inputs = gpt_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = gpt_model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1, pad_token_id=gpt_tokenizer.eos_token_id)
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

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
    # Generate dialogue based on caption
    caption_dialogue = generate_dialogue(f"The image shows {caption}. What do you think?", max_length=50)
    st.write(f"**NeuroLens chats**: {caption_dialogue}")

    # Visual Question Answering
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question (e.g., 'Is the cat black?')")
    if question:
        result = vqa(image, question, top_k=1)
        answer = result[0]["answer"]
        confidence = result[0]["score"]
        st.write(f"**NeuroLens answers**: {answer} ({confidence*100:.1f}% sure)")
        # Generate dialogue based on VQA
        vqa_dialogue = generate_dialogue(f"The answer to '{question}' is '{answer}'. Anything else you'd like to know?", max_length=50)
        st.write(f"**NeuroLens chats**: {vqa_dialogue}")

# Chatbot section
st.subheader("Chat with NeuroLens")
user_input = st.text_input("Say something to NeuroLens (e.g., 'Tell me about cats!')")
if user_input:
    chat_response = generate_dialogue(user_input, max_length=50)
    st.write(f"**NeuroLens chats**: {chat_response}")