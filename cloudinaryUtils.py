import traceback
import cloudinary.api
import cloudinary.uploader
import streamlit as st

from dotenv import dotenv_values, load_dotenv

load_dotenv()

cloudinary.config(
    #cloud_name=st.secrets["REACT_APP_CLOUDINARY_CLOUD_NAME"],
    #api_key=st.secrets["REACT_APP_API_KEY"],
    #api_secret=st.secrets["REACT_APP_API_SECRET"],
    Click 'View Credentials' below to copy your API secret
)

def upload_image_to_cloudinary(image_file)->str|None:
    try:
        upload_result = cloudinary.uploader.upload(image_file)
        image_url = upload_result["secure_url"]
        return image_url
    except Exception as e:
        traceback.print_exc()
        print(f"Error uploading image: {e}")
        return None
