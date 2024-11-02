import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline
import google.generativeai as genai
import os
import time
import base64

os.system('cls')  # Clear the screen on Windows


# Streamlit Page Configuration
st.set_page_config(page_title="Car Repair Chatbot", layout="wide")

# Function to Convert Image to Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your logo
logo_path = "mechanic1.webp"  # Replace with your logo file path

# Convert logo to base64
logo_base64 = get_base64_of_bin_file(logo_path)

# Add custom CSS for sidebar styling and button visibility
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #333333;
        padding: 20px;
        border-radius: 20px;
        margin: 10px;
    }
    [data-testid="stSidebar"] h1 {
        text-align: center;
        color: white;
    }
    button {
        background-color: #FF6666;
        color: white;
    }
    .red-box {
        background-color: #FF4D4D;
        padding: 10px;
        border-radius: 5px;
        color: black;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
    
    
)

# Display the Logo and Title
st.markdown(
    f"""
    <div style='display: flex; align-items: center; justify-content: flex-start; margin-bottom: 40px;'>
        <img src='data:image/png;base64,{logo_base64}' width='50' style='margin-right: 50px;'/>
        <h1 style='display: inline;'>FIXIN.AI</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Greeting text centered
st.markdown(
    """
    <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 50px;'>
        <div style='text-align: center; margin-bottom: 40px;'>
            <h3>Hi Mechanic, What can I do for you today?</h3>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state for search history and selected chat
if 'history' not in st.session_state:
    st.session_state['history'] = {}
if 'selected_chat' not in st.session_state:
    st.session_state['selected_chat'] = None

# Configure the Gemini model with your API key
genai.configure(api_key=os.environ.get("API_KEY", "AIzaSyDWHxkoxLS-MqE-CKqvyogOFP2u6HvPSDc"))

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Load your Stable Diffusion model with optimizations
@st.cache_resource
def load_model():
    image_model_id = "CompVis/stable-diffusion-v1-4"
    image_pipe = StableDiffusionPipeline.from_pretrained(
        image_model_id, 
        revision="fp16",
        torch_dtype=torch.float16
    )
    
    # Use GPU if available, otherwise fallback to CPU
    if torch.cuda.is_available():
        image_pipe = image_pipe.to("cuda")
    else:
        st.warning("CUDA not available. Running on CPU. This will be slower.")
        image_pipe = image_pipe.to("cpu")
    
    return image_pipe

image_pipe = load_model()

# Function to generate a car fix prompt
def generate_car_fix_prompt(problem_description):
    input_prompt = (
        f"Generate a guide for the following car problem, focusing only on actionable repair steps. "
        f"Each step should be concise and clearly require a specific tool or physical interaction with the car.\n"
        f"Ignore any general advice, phrases like 'step by step,' or irrelevant details:\n\n"
        f"Problem: {problem_description}\n\n"
        f"1. Provide only actionable steps that require tools and physical interaction with the car."
    )
def generate_detailed_prompt(user_query):
    if "side mirror" in user_query.lower():
        # Add more details to the prompt
        return f"Step-by-step guide on how to remove the side mirror of a car, including tools needed, specific car model, and safety precautions."
    return user_query



    response = model.generate_content(input_prompt)
    generated_prompt = response.text
    return generated_prompt

# Function to extract actionable steps
def extract_actionable_steps(generated_prompt):
    steps = []
    actionable_keywords = ["remove", "install", "disconnect", "connect", "tighten", "replace", "test"]

    for line in generated_prompt.split('\n'):
        if any(keyword in line.lower() for keyword in actionable_keywords):
            steps.append(line.strip())
    
    return steps

# Function to generate images
def generate_image(prompt, progress_bar, estimated_time):
    try:
        with torch.no_grad():
            num_steps = 25
            for i in range(num_steps):
                time.sleep(estimated_time / num_steps)
                progress_bar.progress((i + 1) / num_steps)
            
            images = image_pipe([prompt], num_inference_steps=num_steps).images[0]
        return images
    except Exception as e:
        st.error(f"An error occurred while generating the image: {e}")
        return None
clear

# Function to watermark images
def watermark_image(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_position = (10, 10)
    draw.text(text_position, text, fill="white", font=font)
    return image

# Function for word-by-word typing effect
def word_by_word_typing(text, delay=0.1):
    placeholder = st.empty()
    words = text.split(' ')
    current_text = ""

    for word in words:
        current_text += word + " "
        placeholder.markdown(current_text.strip())
        time.sleep(delay)

# Sidebar: Search History
st.sidebar.title("Search History")
history_keys = list(st.session_state['history'].keys())

if history_keys:
    st.sidebar.write("Previous Searches:")
    selected_history = st.sidebar.radio("Select a previous search", history_keys)

    # Display previous search results
    if selected_history:
        st.session_state['selected_chat'] = selected_history

if st.session_state['selected_chat']:
    st.sidebar.write(f"Selected Chat: {st.session_state['selected_chat']}")

# Handle user input and generate response
car_problem = st.text_input("Describe your car problem here...")

# Add a "Generate Prompt" button
if st.button("Ask FIXIN.AI"):
    if car_problem:
        with st.spinner("Generating repair guide..."):
            try:
                generated_prompt = generate_car_fix_prompt(car_problem)
                st.subheader("Generated Instructions")
                word_by_word_typing(generated_prompt, delay=0.05)

                actionable_steps = extract_actionable_steps(generated_prompt)

                # Save results to session state for history, including images
                st.session_state['history'][car_problem] = {
                    'prompt': generated_prompt,
                    'steps': actionable_steps,
                    'images': []  # Initialize an empty list to store images
                }

                # Display actionable steps without the image prompts first
                st.subheader("Actionable Steps")
                for i, step in enumerate(actionable_steps):
                    st.write(f"{i + 1}. {step}")

                # Generate images based on the actionable steps
                st.subheader("Generated Images for Each Step")
                for i, step in enumerate(actionable_steps):
                    st.write(f"Generating image for Step {i + 1}: {step}")
                    progress_bar = st.progress(0)
                    image = generate_image(step, progress_bar, estimated_time=0.1 * 25)
                    watermarked_image = watermark_image(image, f"Step {i + 1}")
                    st.session_state['history'][car_problem]['images'].append(watermarked_image)  # Save image to history
                    st.image(watermarked_image, caption=f"Step {i + 1}: {step}", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a car problem description.")

# Display selected chat results if available
if st.session_state['selected_chat']:
    st.subheader(f"Results for: {st.session_state['selected_chat']}")
    selected_result = st.session_state['history'][st.session_state['selected_chat']]
    st.write(f"**Generated Prompt:** {selected_result['prompt']}")
    st.write("**Actionable Steps:**")
    for i, step in enumerate(selected_result['steps']):
        st.write(f"{i + 1}. {step}")

    # Display saved images
    st.subheader("Images for Each Step")
    for i, image in enumerate(selected_result['images']):
        st.image(image, caption=f"Step {i + 1}: {selected_result['steps'][i]}", use_column_width=True)
