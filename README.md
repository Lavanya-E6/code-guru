# 🤖 Code Mentor AI

Code Mentor AI is an AI-powered web application that helps you explain code, fix errors, analyze complexity, and optimize your Python code. It uses Streamlit for the frontend and a local instance of the Hugging Face Transformers library (using PyTorch) for the backend.

## 🛠️ Prerequisites

Before running the application, make sure you have the following installed:
- Python 3.8+
- [Git](https://git-scm.com/)

**Note on GPU support**: This app will automatically use a compatible NVIDIA GPU if available for significantly faster generation. 

## 📦 Setup Instructions

1. **Install dependencies:**
   Open a terminal in this project directory and install the required Python packages by running:

   ```bash
   pip install streamlit torch transformers accelerate
   ```
   *(If you have an NVIDIA GPU, make sure you install the CUDA-enabled version of PyTorch according to the [PyTorch website](https://pytorch.org/get-started/locally/).)*

2. **Run the Application:**
   Start the Streamlit web server:

   ```bash
   streamlit run app.py
   ```

3. **Usage:**
   - The application will open in your default web browser (typically at http://localhost:8501).
   - The AI model (`deepseek-coder-1.3b-instruct`) will be downloaded automatically the first time you run it. This might take a few minutes depending on your internet connection.
   - Paste your Python code into the large editor box.
   - Click one of the four buttons to analyze your code!

---

## 🎨 Features
- **Explain Code**: Line-by-line simple explanations.
- **Fix Errors**: Autodetect syntax or logic errors, highlighting exact corrections. "No errors found" if code is good.
- **Analyze Complexity**: Provides Big-O notation for Time and Space complexity.
- **Optimize Code**: Provides a cleaner, faster version of your logic.
- **Dark Theme**: Fully styled with an integrated code editor look.
