import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from PIL import Image

def compress_channel(channel, rank):
    # Perform SVD compression on a single channel
    U, D, V = np.linalg.svd(channel, full_matrices=False) # Singular value Decomposition function
    return np.dot(U[:, :rank], np.dot(np.diag(D[:rank]), V[:rank, :]))
    # image = UDV^(T) - (T) denotes transpose of the matrix

def compress_image(img, rank, is_gray):
    # Compress an image using SVD with a given rank
    img = img.astype(np.float32)
    
    if is_gray:
        compressed = compress_channel(img, rank)
        compressed = np.clip(compressed, 0, 255).astype(np.uint8)
        return compressed
    
    B, G, R = cv2.split(img)
    B_compressed = compress_channel(B, rank)
    G_compressed = compress_channel(G, rank)
    R_compressed = compress_channel(R, rank)
    
    B_compressed = np.clip(B_compressed, 0, 255).astype(np.uint8)
    G_compressed = np.clip(G_compressed, 0, 255).astype(np.uint8)
    R_compressed = np.clip(R_compressed, 0, 255).astype(np.uint8)
    # Merge all the channels of the color image
    return cv2.merge([B_compressed, G_compressed, R_compressed])

st.title("SVD Image Compression")
st.write("Upload an image, specify a rank, and download the compressed image.")

# Upload the file using st.file_uploader function 
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])
# If the uploaded image is in the correct format 
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    is_gray = len(img_array.shape) == 2  # Check if the image is grayscale
    if not is_gray: # If it is a color Image convert RGB to BGR image
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    min_dim = min(img_array.shape[:2]) # Convert the image to square image
    # A rank slider to select the Compression Rank
    rank = st.slider("Select Compression Rank", 1, min_dim-1, min_dim//2)
    
    if st.button("Compress Image"):
        compressed_img = compress_image(img_array, rank, is_gray)
        # st.image to display the compressed image (Color or Gray)
        st.image(compressed_img, caption="Compressed Image", use_container_width=True, channels="RGB" if not is_gray else "GRAY")
        # Encode the image to Joint Photograpgic Expert Group
        _, buffer = cv2.imencode(".jpg", compressed_img)
        compressed_bytes = BytesIO(buffer)
        # Streamlit Download button to Download the compressed image
        st.download_button(
            label="Download Compressed Image",
            data=compressed_bytes,
            file_name="compressed_image.jpg",
            mime="image/jpeg"
        )
