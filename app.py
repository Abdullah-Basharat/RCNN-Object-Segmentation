import streamlit as st
from maskrcnn import load_model, predict_image, visualize_results

model = load_model()

# Streamlit UI
st.title("Mask RCNN Object Segmentation")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)

if uploaded_file is not None:
    with st.spinner("Processing..."):
        image_path = f"uploaded_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        original_image, boxes, masks = predict_image(model, image_path, threshold)
        result_image = visualize_results(original_image.copy(), boxes, masks)

        # Display Results
        st.image(original_image, caption="Original Image", use_container_width=True)
        st.image(result_image, caption="Segmented Image", use_container_width=True)
