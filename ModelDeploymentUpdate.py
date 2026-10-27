import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from tensorflow.keras import backend as K
from huggingface_hub import hf_hub_download
import os
from datetime import datetime
import streamlit.components.v1 as components
from openpyxl import load_workbook
import io
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account



# Set the title of the web app
st.set_page_config(page_title="Flourish - Vegetation Analysis", layout="wide")

# Model paths
#model_paths = {
 #   'U-Net': r"C:\Users\R\OneDrive\Desktop\Senior Project\unet_model(f).h5",
  #  'DeepLabV3+': r"C:\Users\R\OneDrive\Desktop\Senior Project\Fdeeplabv3+_model.keras",
   # 'PSPNet': r"C:\Users\R\OneDrive\Desktop\Senior Project\Fpspnet_model.keras",
#}

# Each model: (Hugging Face repo, file name inside that repo)
model_files = {
    'U-Net': ("AlghamdiNouraSH/UNET_Model.h5", "UNET_Model.h5"),
    'DeepLabV3+': ("AlghamdiNouraSH/Fdeeplabv3plus_model.keras", "Fdeeplabv3plus_model.keras"),
    'PSPNet': ("AlghamdiNouraSH/Fpspnet_model.keras", "Fpspnet_model.keras"),
}

# Function to load selected model
@st.cache_resource(max_entries=1)
def load_selected_model(selected_model):
    repo_id, filename = model_files[selected_model]
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_model(
        path,
        custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou_metric': iou_metric},
        compile=False
    )

#############



# Function to load selected model
#def load_selected_model(selected_model):
 #   return load_model(
  #      model_paths[selected_model],
   #     custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'iou_metric': iou_metric},
    #    compile=False
    #)

# Metrics for custom loss functions
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), tf.float32)
    y_pred_f = K.cast(K.flatten(y_pred), tf.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1]) 
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Image preprocessing 
def preprocess_image(image):
    img = tf.image.convert_image_dtype(image, tf.float32)
    img = tf.image.resize(img, (256, 256))
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function to generate green/black mask
def predict(image, model):
    processed_image = preprocess_image(tf.convert_to_tensor(np.array(image)))
    prediction = model.predict(processed_image)
    prediction = np.squeeze(prediction)  # Remove batch dimension

    # Create a color mask :
    # 1 (greenery) -> [0, 255, 0] (green)
    # 0 (non-greenery) -> [0, 0, 0] (black)
    color_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    color_mask[prediction > 0.5] = [0, 255, 0]  # Green for greenery
    color_mask[prediction <= 0.5] = [0, 0, 0]   # Black for non greenery

    return color_mask

# Streamlit page setup
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Main page 
def home_page():
    st.markdown("""<style>
        body {
            background-color: #E8F5E9;  /* Light Green  */
            font-family: 'Roboto', sans-serif;
            color: #333333;
        }
        .header {
            text-align: center;
            padding: 20px;
            color: #2E7D32;  /* Darker Green */
            font-size: 2.5em;
        }
        .sub-header {
            text-align: center;
            font-size: 1.5em;
            margin-bottom: 40px;
            color: #388E3C;  /* Medium Green */
        }
        .feature-card {
            background-color: #FFFFFF;  /* White Background */
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            color: #333333;
            height: 170px;
        }
        .feature-card:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);  /* Shadow */
        }
        .highlight {
            color: #FF5722;  /* Highlight Color */
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Flourish")
    st.markdown("<h2 class='sub-header'>Analyze Vegetation Levels with Flourish</h2>", unsafe_allow_html=True)
    #st.markdown("<p style='text-align: center; font-size: 18px;'>Flourish is a tool for evaluating vegetation levels, providing visual insights and data on greenery to aid environmental monitoring and landscape planning.</p>", unsafe_allow_html=True)

    # cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class='feature-card'style='background-color: #E8F5E9;'>  <!-- Light Green -->
                <h3 style='color: #2E7D32;'>Image Prediction🌿</h3>  <!-- Dark Green  -->
                <p>Upload images and calculate the Green View Index (GVI) to assess vegetation using a Deep learning Model of your choice out of Three Models.</p>
            </div>""", unsafe_allow_html=True)
        if st.button("Prediction", key="pred_button", use_container_width=True):
            st.session_state.page = "Prediction"

    with col2:
        st.markdown("""<div class='feature-card' style='background-color: #E8F5E9;'>  <!-- Light Green -->
                <h3 style='color: #2E7D32;'>Data Visualization📊</h3>  <!-- Dark Green  -->
                <p>View Greenery Levels on an Interactive Dashboard of Various Locations in Makkh Province</p>
            </div>""", unsafe_allow_html=True)
        if st.button("Visualization", key="vis_button", use_container_width=True):
            st.session_state.page = "Visualization"

    #with col3:
     #   st.markdown("""<div class='feature-card' style='background-color: #E8F5E9;'>  <!-- Light Green  -->
      #          <h3 style='color: #2E7D32;'>Add New Location📍</h3>  <!-- Dark Green  -->
       #         <p>Input a new location's details, upload an image, and calculate the GVI.</p>
        #    </div>""", unsafe_allow_html=True)
        #if st.button("Add Location", key="add_location_button", use_container_width=True):
         #   st.session_state.page = "Add Location"

def prediction_page():
    if st.button("< Back"):
        st.session_state.page = "Home"
        
    st.title("Image Prediction")
    st.write("Predict and Calculate the Green View Index (GVI) from Images")
    
    # Slider to select the number of images to upload
    num_images = st.slider("Select the number of images to upload:", min_value=1, max_value=10, value=4)

    st.markdown(f"<p class='highlight'>To calculate the Green View Index (GVI), please upload <strong>{num_images} images</strong> of the same location taken from different directions.</p>", unsafe_allow_html=True)

    selected_model = st.selectbox("Select Model:", list(model_files.keys()))
    with st.spinner(f"Loading {selected_model}..."):
        model = load_selected_model(selected_model)
   
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) != num_images:
            st.error(f"Please upload exactly {num_images} images.")
        else:
            cols = st.columns(min(num_images, 4)) 
            total_area = 0
            total_greenery_area = 0

            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                
                with cols[i % 4]:  
                    st.image(image.resize((150, 150)), caption="Original Image", use_column_width=True)  
                    prediction_image = Image.fromarray(predict(image, model))  # Use the color mask
                    st.image(prediction_image.resize((150, 150)), caption="Prediction Image", use_column_width=True, clamp=True)

                    total_image_area = prediction_image.size[0] * prediction_image.size[1]
                    greenery_area = np.sum(np.array(prediction_image)[:, :, 1] > 0)  # Count green pixels

                    total_area += total_image_area
                    total_greenery_area += greenery_area

            if total_area > 0:
                gvi = total_greenery_area / total_area
                gvi_percentage = gvi * 100  

                st.markdown(f"""
                <div style="padding: 10px; border: 2px solid green; background-color: #E8F5E9; border-radius: 10px; text-align: center;">
                    <h2 style="color: #388E3C;">Green View Index (GVI): {gvi_percentage:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("No data available to calculate GVI.")


def visualization_page():
    if st.button("< Back"):
        st.session_state.page = "Home"
        
    st.title("Data Visualization")

    
    # Embed Power BI report
    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiMjJiYTIzMzMtN2UwYS00MjRmLTg0ZTgtYzQ2ZWY3Nzc3Y2JlIiwidCI6ImI0NTNkOTFiLTZhYzEtNGI2MS1iOGI4LTVlNjVlNDIyMjMzZiIsImMiOjl9"
    
    
    components.html(
        f"""
        <iframe title="Power BI Report" width="100%" height="800px" src="{power_bi_url}" frameborder="0" allowFullScreen="false"></iframe>
        """,
        height=800,
        scrolling=True
    )


SERVICE_ACCOUNT_JSON = {

}
SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = '1yXd5ux7gfioxjwpGSxHfAOPWpnuINszA'  


def upload_to_drive(file_stream, file_name):
    credentials = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)

    file_metadata = {
        'name': file_name,
        'parents': [FOLDER_ID]
    }
    media = MediaIoBaseUpload(file_stream, mimetype='image/jpeg', resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    file_id = file.get('id')
    # thumbnail URL
    return f'https://drive.google.com/thumbnail?id={file_id}'



def add_location_page():
    if st.button("< Back"):
        st.session_state.page = "Home"

    st.title("Add New Location")
    st.write("Input location details and upload 4 images in the same location taken from different directions to add a new location.")

    # Input form
    with st.form("Add Location Form"):
        latitude = st.number_input("Latitude", format="%.6f")
        longitude = st.number_input("Longitude", format="%.6f")
        province = st.text_input("Province")

        #uploader for exactly 4 images
        uploaded_images = st.file_uploader("Choose 4 images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        selected_model = st.selectbox("Select Model:", list(model_paths.keys()))
        submitted = st.form_submit_button("Add and Predict GVI")

        if submitted:
            
            if not province:
                st.error("Please enter a valid province.")
            elif not uploaded_images or len(uploaded_images) != 4:
                st.error("Please upload exactly 4 images.")
            else:
                model = load_selected_model(selected_model)
                total_image_area = 0
                total_greenery_area = 0
                image_urls = []

                # Get date
                current_date = datetime.now().strftime('%Y-%m')

                for i, uploaded_image in enumerate(uploaded_images):
                    image = Image.open(uploaded_image)

                    # Format the filename
                    image_filename = f"image_{str(latitude).replace('.', '-')}_{str(longitude).replace('.', '-')}_{i}_{current_date}.jpg"

                    file_stream = io.BytesIO()
                    image.save(file_stream, format='JPEG')
                    file_stream.seek(0)  

                    # Upload image to Drive and get the URL
                    image_url = upload_to_drive(file_stream, image_filename)
                    image_urls.append(image_url)

                    # Predict for each image
                    prediction_image = predict(image, model)
                    image_area = prediction_image.shape[0] * prediction_image.shape[1]
                    greenery_area = np.sum(prediction_image[:, :, 1] > 0)
                        
                    total_image_area += image_area
                    total_greenery_area += greenery_area
                    
                # Calculate GVI
                if total_image_area > 0:
                    gvi = total_greenery_area / total_image_area
                    gvi_ratio = (total_greenery_area / total_image_area) * 100
                    format_gvi_ratio = f"{gvi_ratio:.2f}%"
                else:
                    st.error("No valid image area found.")
                    return

                # greenery level
                if gvi_ratio < 5:
                    greenery_level = "Very Low"
                elif gvi_ratio < 15:
                    greenery_level = "Low"
                elif gvi_ratio < 25:
                    greenery_level = "Normal"
                elif gvi_ratio < 35:
                    greenery_level = "High"
                else:
                    greenery_level = "Very High"

                # Display GVI result
                st.markdown(f"""
                <div style="padding: 10px; border: 2px solid green; background-color: #E8F5E9; border-radius: 10px; text-align: center;">
                    <h2 style="color: #388E3C;">Green View Index (GVI): {format_gvi_ratio}</h2>
                </div>
                """, unsafe_allow_html=True)

                # Append data to Excel 
                new_data = pd.DataFrame({
                    "Latitude": [latitude],
                    "Longitude": [longitude],
                    "Date": [current_date],
                    "Total_Greenery_Area": [total_greenery_area],
                    "Total_Area": [total_image_area],
                    "GVI": [gvi],
                    "GVI Ratio": [format_gvi_ratio],
                    "Greenery Level": [greenery_level],
                    "Province": [province],
                    "Image URL 1": [image_urls[0] if len(image_urls) > 0 else None],
                    "Image URL 2": [image_urls[1] if len(image_urls) > 1 else None],
                    "Image URL 3": [image_urls[2] if len(image_urls) > 2 else None],
                    "Image URL 4": [image_urls[3] if len(image_urls) > 3 else None],
                })
                excel_file_path = r"C:\Users\R\OneDrive\Desktop\Final_model\GVI_resultf.xlsx"

                if os.path.exists(excel_file_path):
                    with pd.ExcelWriter(excel_file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                        new_data.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
                else:
                    with pd.ExcelWriter(excel_file_path, mode='w', engine='openpyxl') as writer:
                        new_data.to_excel(writer, index=False)

                st.success("New location added successfully.")




# Main page 
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Prediction":
    prediction_page()
elif st.session_state.page == "Visualization":
    visualization_page()
elif st.session_state.page == "Add Location":
    add_location_page()
