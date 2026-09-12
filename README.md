# Suryavision - Deep Learning Healthcare

An AI-powered dermatological diagnostic tool that analyzes skin lesions using state-of-the-art Deep Learning models. 

## Features
- **Semantic Segmentation:** Uses U-Net and Self-Attention U-Net to precisely identify and mask the boundaries of a skin lesion.
- **Disease Classification:** Uses a MobileNetV2 architecture trained on dermoscopic datasets to predict the probability of Melanoma, Common Nevi, and Atypical Nevi.
- **Smart Image Validation:** Automatically detects and rejects non-skin images (like random objects or scenery) to prevent false positives.
- **Premium Interface:** A modern, glassmorphism-styled dark mode UI for a professional healthcare experience.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/suryavision.git
   cd suryavision
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server:**
   ```bash
   cd backend
   uvicorn app:app --port 8081 --host 127.0.0.1
   ```

4. **Launch the frontend:**
   Open `frontend/index.html` in your web browser. Or serve it using python:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   Then navigate to `http://localhost:3000`

## Architecture
- **Backend:** FastAPI (Python), PyTorch, Torchvision
- **Frontend:** HTML, CSS (Glassmorphism), Vanilla JavaScript
- **Models:** Standard U-Net, Self-Attention U-Net, MobileNetV2 (Classifier)
