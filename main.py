import streamlit as st
import tensorflow as tf
import numpy as np
import base64

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element



st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Public+Sans:ital,wght@0,100..900;1,100..900&display=swap');
    
    html, body, [class*="stApp"]  {
        background-color: #0e0e0e;
        color: #FFFFFF;
        font-family: 'Public Sans', sans-serif;
    } 
    .main-title {
        font-size: 36px;
        font-weight: 900;
        color: #ffffff;
    }
    .highlight {
        color: #34eb5c;
    }
    h2 {
        color: #ffffff;  /* This will make subheadings white */
    }
    p{
        color: #8F9093;         
    }
    .file-upload {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px solid #34eb5c;
    }
    .predict-btn {
        background-color: #34eb5c;
        color: #000000;
        font-size: 20px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 7px 30px;
        cursor: pointer;
        display: inline-block;
    }
    .predict-btn:hover {
        background-color: #28c64b;  /* Slightly darker green for hover effect */
    }
    .result-box {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        text-align: center;
        color: #34eb5c;
    }
    .image-container {
        background-color: #1a1a1a;
        border-radius: 20px;
        padding: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px solid #2e2e2e;
        width: 80%;
        margin: 0 auto;
        height: 400px;  /* Adjust this based on the desired size */
    }
    img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 15px;
    }
    /* Make the layout responsive */
    @media only screen and (max-width: 768px) {
        .main-title {
            font-size: 28px;
        }
        .file-upload {
            flex-direction: column;
        }
        .predict-btn {
            font-size: 16px;
            padding: 8px 16px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<h1 class="main-title">Low-Cost Accessible Plant <span class="highlight">Disease Detection</span></h1>', unsafe_allow_html=True)
st.write("Introducing a low-cost plant disease detection system that leverages computer vision technology to provide small-scale farmers with a practical and affordable solution to one of their major challenges: crop health monitoring.")

# File upload section
uploaded_file = st.file_uploader("Upload file here", type=["jpg", "jpeg", "png"])



# Display the uploaded image below the file uploader
# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if uploaded_file is not None:
    # Encode the uploaded image as base64
    encoded_image = base64.b64encode(uploaded_file.read()).decode('utf-8')
    
    st.markdown(f'<div class="image-container"><img src="data:image/png;base64,{encoded_image}"/></div>', unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)

# Button in Streamlit using custom styling
if(st.button("Predict")):
    st.write("Our Prediction")

    with st.spinner('Wait a moment ...'):
        if uploaded_file is not None:
            result_index = model_prediction(uploaded_file)

            #Reading labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
            st.success("Model is Predicting it's a {}".format(class_name[result_index]))

    #Reading Labels

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


   
# if st.markdown('<button class="predict-btn">Predict</button>', unsafe_allow_html=True):
    
#     st.write("Prediction in progress...")
#     if uploaded_file is not None:
#         result_index = model_prediction(uploaded_file)

#         #Reading labels
#         class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
#                 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
#                 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
#                 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
#                 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#                 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
#                 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
#                 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
#                 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
#                 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
#                 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                     'Tomato___healthy']
#         st.success("Model is Predicting it's a {}".format(class_name[result_index]))



# How it helps small farmers section with updated white subheading
st.markdown("""
    ## How It Would Help Small Farmers
    1. **Early Disease Detection:** Early identification of diseases reduces crop losses and helps farmers save on unnecessary chemical applications.
    2. **Increased Productivity:** By reducing the spread of diseases, this tool can help farmers improve their yields, allowing them to remain competitive in the market.
    3. **Cost Savings on Labor:** Automating disease detection saves labor costs, making it a cost-effective solution.
    4. **Scalability of the Solution:** Small farms can start using the app with a simple smartphone submission system and scale up as they grow.
    5. **Accessible for Niche Markets:** Helps small farmers meet organic or specialty market standards without relying on chemicals.
""")

# Economic Impact and Sustainability Section
st.markdown("""
    ## Economic Impact and Support for Sustainability
    This tool helps small farms remain economically viable by reducing the need for chemical inputs and improving crop health.
""")

# Reputable Support section
st.markdown("""
    ## Reputable Support
    This approach aligns with broader goals of sustainable agriculture and USDA initiatives to support small farmers through technology.
""")