# smart_meter_ocr_advanced.py
import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import tempfile

st.set_page_config(page_title="EBS Meter OCR", page_icon="⚡")
st.title("⚡ EBS Meter OCR - Self Billing ")
# st.write("Uploads tilted or reflective meter images and accurately extracts digital readings (supports decimals).")

# Optional: path to tesseract if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

uploaded_file = st.file_uploader("📸 Upload Meter Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        img_path = tmp_file.name

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 1️⃣ Detect display region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display_region = None
    best_cnt = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        if 10000 < area < 450000 and 2 < aspect_ratio < 8:
            display_region = image[y:y+h, x:x+w]
            best_cnt = cnt
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break

    st.image(image_rgb, caption="Detected Display Region", use_container_width=True)

    if display_region is not None:
        # Step 2️⃣ Correct rotation using minimum area rectangle
        rect = cv2.minAreaRect(best_cnt)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle

        (h, w) = display_region.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(display_region, rot_matrix, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Step 3️⃣ Preprocessing for OCR
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Adaptive threshold (invert for light background)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 15
        )

        # Dilate to join 7-segment lines
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.dilate(adaptive, kernel, iterations=1)

        # Resize and smooth
        resized = cv2.resize(morph, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        resized = cv2.medianBlur(resized, 3)

        st.image(resized, caption="Enhanced Display (Preprocessed)", channels="GRAY", use_container_width=True)

        # Step 4️⃣ OCR with digits + dot support
        # Use whitelist to include digits and decimal point
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
        raw_text = pytesseract.image_to_string(resized, config=custom_config)

        # Extract numeric reading including decimals
        matches = re.findall(r'\d+\.\d+|\d+', raw_text)
        reading = max(matches, key=len) if matches else None

        st.subheader("📋 OCR Raw Output:")
        st.text(raw_text.strip())

        if reading:
            st.success(f"✅ Detected Meter Reading: **{reading}**")
        else:
            st.error("❌ Could not detect digits clearly. Try adjusting focus or lighting.")
    else:
        st.warning("⚠️ Display region not detected. Try uploading a front-facing, clear photo.")
