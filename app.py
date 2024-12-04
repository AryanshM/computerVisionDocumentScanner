import streamlit as st
import cv2
import numpy as np
import os
import utlis
import easyocr
import re
import time
from PIL import Image

import cv2
import numpy as np
import os
import easyocr

def warpAndScan(image, output_dir="./filteredImages/"):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {image}. Please check the file path or its integrity.")

    print("File successfully read")

    # Define dimensions for processing and original resolutions
    h, w = 1600, 1200
    heightImg, widthImg = 800, 600

    # Resize for processing
    original_image = img.copy()
    img = cv2.resize(img, (widthImg, heightImg))
    original_image = cv2.resize(original_image, (w, h))

    # Preprocessing
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = utlis.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Find contours
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxArea = utlis.biggestContour(contours)

    # Initialize text
    extracted_text = ""

    if biggest.size != 0:
        print("Processing the biggest contour")

        # Scale contour points for original resolution
        biggest_original = np.copy(biggest)
        for i in range(4):
            biggest_original[i][0][0] *= int(w / widthImg)
            biggest_original[i][0][1] *= int(h / heightImg)

        # Reorder points for perspective transformation
        biggest = utlis.reorder(biggest)
        biggest_original = utlis.reorder(biggest_original)

        # Perspective transformation matrices
        pts1 = np.float32(biggest)
        pts1_original = np.float32(biggest_original)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        pts2_original = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        matrix_original = cv2.getPerspectiveTransform(pts1_original, pts2_original)

        # Warping
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg), flags=cv2.INTER_CUBIC)
        imgWarpColored_original = cv2.warpPerspective(original_image, matrix_original, (w, h), flags=cv2.INTER_CUBIC)
        imgWarpColored_original = cv2.resize(imgWarpColored_original, (1300, 840))

        # Save processed images
        images = [
            ("0imgBlank.jpg", imgBlank),
            ("1imgGray.jpg", imgGray),
            ("2imgBlur.jpg", imgBlur),
            ("3imgThreshold.jpg", imgThreshold),
            ("4imgContours.jpg", img),
            ("5imgBigContour.jpg", img),
            ("6imgWarpColored.jpg", imgWarpColored),
            ("7imgWarpColoredOriginal.jpg", imgWarpColored_original),
        ]

        for name, image in images:
            cv2.imwrite(os.path.join(output_dir, name), image)

        # OCR to extract text
        print("Starting OCR process...")
        reader = easyocr.Reader(['en', 'hi'])
        result = reader.readtext(imgWarpColored_original)
        print("OCR process completed.")
        extracted_text = ' '.join([entry[1] for entry in result])
    else:
        print("No significant contours found!")
    print(extracted_text)
    return [os.path.join(output_dir, name) for name, _ in images], extracted_text

def frontOrBack(text):
    keywords = ["Address", "A dd ress", "Addrèss", "Addre$$", "sserdA"]
    return "back" if any(k.lower() in text.lower() for k in keywords) else "front"

def extractFront(text):
    # searching for name in front 
    start = text.find("DO8")
    if(start==-1):
        start = text.find("D08")
    if(start==-1):
        start = text.find("D08")
    if(start==-1):
        start = text.find("D0B")
    if(start==-1):
        start = text.find("DOB")

    if start == -1:
        return None, None, None

    # Name extraction
    name = ""
    english = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    first_instance = second_instance = -1
    for i in range(start - 1, -1, -1):
        if text[i] in english:
            second_instance = i
            break
    for i in range(i - 1, -1, -1):
        if text[i] not in english:
            if not text[i].isspace():
                first_instance = i + 1
                break
    name = text[first_instance:second_instance + 1]

    # DOB extraction
    dob_pattern = r"(\d{2}/\d{2}/\d{4})"
    dob_match = re.search(dob_pattern, text)
    dob = dob_match.group(1) if dob_match else None

    # ID extraction
    aadhar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    aadhar_match = re.search(aadhar_pattern, text)
    id_number = aadhar_match.group() if aadhar_match else None
    print(name, dob, id_number)

    return name, dob, id_number

def extractBack(text):
    address_start = text.lower().find("address:")
    if address_start == -1:
        return None
    return text[address_start + len("address:"):].strip()

def display_images_sequentially(images):
    """
    Display each image in the list for 1 second and then replace it with the next image.
    """
    placeholder = st.empty()  # Create a placeholder for displaying images
    for img_path in images:
        img = Image.open(img_path)
        placeholder.image(img, use_column_width=True)
        time.sleep(1)

# Streamlit Application
st.title("Image Processing, OCR, and Information Extraction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("Image uploaded successfully!")

    try:
        # Process the image
        output_dir = "./filteredImages/"
        processed_images, extracted_text = warpAndScan(temp_path, output_dir)

        # Determine front or back and extract details
        side = frontOrBack(extracted_text)
        print("side : ",side)
        if side == "front":
            name, dob, id_number = extractFront(extracted_text)
            print(name, dob, id_number)
            address = None
        else:
            name, dob, id_number = None, None, None
            address = extractBack(extracted_text)
            print(address)

        # Display processed images one by one
        st.write("Displaying processed images sequentially:")
        display_images_sequentially(processed_images)

        # Display extracted information
        st.write("### Extracted and Filtered Information:")
        if side == "front":
            st.write(f"**Name**: {name if name else 'Not Found'}")
            st.write(f"**Date of Birth**: {dob if dob else 'Not Found'}")
            st.write(f"**ID Number**: {id_number if id_number else 'Not Found'}")
        else:
            st.write(f"**Address**: {address if address else 'Not Found'}")

        # Display full extracted text
        st.write("### Full Extracted Text:")
        st.text(extracted_text)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
