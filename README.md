# FIXIN.AI

INTRODUCTION:

  FIXIN.AI is an installation guidance chatbot designed to assist mechanics with step-by-step instructions for car part repairs and installations. By leveraging AI-driven text and image generation, FIXIN.AI provides mechanics with concise instructions and visual guides that improve efficiency and accuracy in repair tasks. This tool aims to streamline the repair process, reduce error rates, and facilitate real-time support in the workshop.

REQUIREMENTS:

To run the FIXIN.AI application, ensure you have the following libraries and tools installed:

Streamlit: For building the web interface

Python: Core language for backend development

Torch: Required for model handling

CSS and HTML: For frontend styling

Diffusers: For image generation via Stable Diffusion

PIL: For image processing

google-generativeai: Integration with Google's Gemini model for text generation

os, time, base64: System and utility libraries for various functionalities

MODEL USED:

Text Generation: Utilizes Google's Generative AI (Gemini Model) to generate descriptive repair instructions based on user input.

Image Generation: Employs the Stable Diffusion Pipeline to create visual representations for each repair step. This visual aid enhances comprehension and makes complex procedures more accessible.

Session Management: Tracks user interactions, maintaining a history of past repair instructions for easy access and reference.

WORKING:

FIXIN.AI operates through a sequence of modules that interact seamlessly:

User Interface Module: Built with Streamlit, this module allows mechanics to enter a problem description via text or voice input.

Prompt Engineering Module: This module formulates user prompts to generate accurate and contextually relevant outputs for car repair guidance.

Text Generation Module: Using Gemini, it translates problem descriptions into actionable repair steps.

Image Generation Module: Stable Diffusion generates images that align with each step, providing visual aids alongside the textual guide.

Image Processing Module: Enhances generated images with labels, watermarks, and grid layouts for structured viewing.

Session Management Module: Maintains a search history to facilitate easy access to previous repair sessions.

FEATURES:

Real-Time Interaction: Mechanics can receive immediate feedback on repair steps through a user-friendly interface.

Text and Image-Based Guidance: Instructions are supplemented with generated images to improve clarity.

Voice Input: For convenience, FIXIN.AI allows mechanics to dictate their queries.

Session History: Previous interactions are saved, allowing users to revisit past repairs and troubleshoot effectively.

CONCLUSION:

  FIXIN.AI represents a step forward in workshop automation, providing mechanics with an AI-powered assistant to improve efficiency and accuracy in car part repairs. Through a combination of advanced text and image generation models, FIXIN.AI is poised to become an invaluable tool for modern repair shops.


