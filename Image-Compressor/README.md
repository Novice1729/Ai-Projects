**README.md**

# SVD Image Compression using Streamlit

This project implements **Singular Value Decomposition (SVD)** to compress images using **Streamlit** for the user interface. Users can upload an image, specify the desired rank, and download the compressed image.

## Features
- Supports **both color and grayscale images**
- Interactive **slider to select compression rank**
- Displays **compressed image** after processing
- Allows **downloading the compressed image**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/svd-image-compression.git
   cd svd-image-compression
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dependencies
See `requirements.txt` for the list of required Python libraries.

## Usage
- Upload an image (JPEG, PNG, BMP)
- Choose the compression rank using the slider
- View the compressed image
- Download the compressed image

## License
MIT License

---
