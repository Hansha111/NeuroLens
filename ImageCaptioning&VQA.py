# -*- coding: utf-8 -*-
"""NeuroLens.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZqgfOyvWTqdG_rjajatqNLjjFuX7P1o9
"""

!pip install transformers torch PIL

!pip install pillow

from PIL import Image
print("Pillow is working!")

from transformers import pipeline
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
print("BLIP is ready to describe pictures!")

from PIL import Image
import requests
from io import BytesIO

# Get the image from the internet
image_url = "https://plus.unsplash.com/premium_photo-1738854512331-3ed5ec9cf56a?q=80&w=3135&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Use BLIP to describe the picture
caption = captioner(image)[0]["generated_text"]
print("The picture shows:", caption)

from google.colab import files
uploaded = files.upload()

from google.colab import files
files.download("cat.jpg")

from PIL import Image

# Load the uploaded image
image_file = "sample_img.jpg"
image = Image.open(image_file)

# Use BLIP to describe the picture
caption = captioner(image)[0]["generated_text"]
print("The picture shows:", caption)

!pip install gradio

from transformers import pipeline

# Load the BLIP-VQA model
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
print("BLIP-VQA is ready to answer questions!")

from PIL import Image
import requests
from io import BytesIO

# Get the image from the internet
image_url = "https://plus.unsplash.com/premium_photo-1738854512331-3ed5ec9cf56a?q=80&w=3135&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"  # Horse image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Ask a question about the image
question = "What is the color of the lady?"
answer = vqa_pipeline(image, question, top_k=1)[0]["answer"]
print("Question:", question)
print("Answer:", answer)

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
from google.colab import files

# Load the VQA model (like a superhero who answers questions about pictures)
vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")

# Welcome message
print("Welcome to NeuroLens VQA! Let's look at a picture and answer your questions!")

# Ask user how they want to provide the image
image_choice = input("Do you want to upload an image (type 'upload') or use a URL (type 'url')? ")

if image_choice.lower() == "upload":
    # Let user upload an image
    print("Please upload your image (jpg or png).")
    uploaded = files.upload()
    image_name = list(uploaded.keys())[0]  # Get the uploaded file name
    image = Image.open(BytesIO(uploaded[image_name]))  # Open the image
else:
    # Get image from URL
    image_url = input("Enter the image URL (e.g., https://example.com/dog.jpg): ")
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))  # Open the image from URL

# Show a fun message
print("Got the image! Now, ask a question about it.")

# Get the user's question
question = input("What do you want to know about the image? (e.g., 'Is the dog brown?') ")

# Use the VQA model to get the answer
result = vqa(image, question, top_k=1)

# Print the answer in a fun way
answer = result[0]["answer"]
confidence = result[0]["score"]
print(f"NeuroLens says: {answer}! (I'm {confidence*100:.1f}% sure!)")

# Save the image for later (optional)
image.save("last_image.png")
print("Image saved as 'last_image.png' in Colab. You can download it!")

