import json
import os
import time
import traceback

import cv2
import base64
from PIL import Image

import re
import numpy as np
import pandas as pd

import requests
import streamlit as st
from google.cloud import vision_v1p3beta1 as vision

from cropImg import extractPersonImg
from cloudinaryUtils import upload_image_to_cloudinary

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    r"./tactile-sentry-378309-9645637b7151.json"
)

allRes = {}

allData = {}


def detect_handwritten_ocr(path)-> tuple[str,list]:
    """
    Detects handwritten characters in a local image.

    Args:
    path: The path to the local file.
    """
    client = vision.ImageAnnotatorClient()

    content = path
    image = vision.Image(content=content)

    image_context = vision.ImageContext(language_hints=["en-t-i0-handwrit"])
    response = client.document_text_detection(image=image, image_context=image_context) # type: ignore

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
        
        
    markings = []

    for composite_field in response.text_annotations:
        description = str(composite_field.description)
        vertices = [(composite_field.bounding_poly.vertices[i].x,
                    composite_field.bounding_poly.vertices[i].y)
                    for i in range(4)]
        markings.append({
            "description":description,
            "vertices":vertices
        })
        
    

    return response.full_text_annotation.text, markings


def parser_text(dd:str, selection:str) -> dict[str, str]:
    if selection=="HORIZON_ACADEMY":
        name = re.findall(r"Name:\s*(.*?)\s*DOB:", dd)
        DOB = re.findall(r"DOB:\s*(.*?)\s*Gender", dd)
        Gender = re.findall(r"Gender\s*(.*?)\s*Phone", dd)
        Phno = re.findall(r"no:\s*(.*?)\s*Activity", dd)
        Activity = re.findall(r"Activity:\s*(.*?)\s*Height", dd)
        Height = re.findall(r"Height\s*(.*?)\s*Weight", dd)
        Weight = re.findall(r"Weight\s*(.*?)\s*Health", dd)
        HealthIssues = re.findall(r"Health Issues\s*(.*?)\s*Father", dd)
        Father = re.findall(r"Father name:\s*(.*?)\s*Mother", dd)
        Mother = re.findall(r"Mother Name:\s*(.*?)\s*Current", dd)
        try:
            CurrentAddress = [re.search(r"Current Address:(.*?)City", dd, re.DOTALL).group(1).strip()] # type: ignore
        except:
            CurrentAddress = [""]
        City = re.findall(r"City\s*(.*?)\s*State", dd)
        State = re.findall(r"State\s*(.*?)\s*Pincode", dd)
        Pincode = re.findall(r"Pincode\s*(.*?)\s*Email ID:", dd)
        EmailID = re.findall(r"Email ID:\s*(.*?)\s*Guardian Name:", dd)
        GuardianName = re.findall(r"Guardian Name:\s*(.*?)\s*Coach Name:", dd)
        CoachName = re.findall(r"Coach Name:\s*(.*?)\s*Date", dd)
        registrationDate = re.findall(r"Date\s*(.*?)\s*Parent/Guardian Name", dd)

        return {
            "name": name[0] if len(name) > 0 else "",
            "DOB": DOB[0] if len(DOB) > 0 else "",
            "Gender": Gender[0] if len(Gender) > 0 else "",
            "Phno": Phno[0] if len(Phno) > 0 else "",
            "Activity": Activity[0] if len(Activity) > 0 else "",
            "Height": Height[0] if len(Height) > 0 else "",
            "Weight": Weight[0] if len(Weight) > 0 else "",
            "HealthIssues": HealthIssues[0] if len(HealthIssues) > 0 else "",
            "Father": Father[0] if len(Father) > 0 else "",
            "Mother": Mother[0] if len(Mother) > 0 else "",
            "CurrentAddress": CurrentAddress[0] if len(CurrentAddress) > 0 else "",
            "City": City[0] if len(City) > 0 else "",
            "State": State[0] if len(State) > 0 else "",
            "Pincode": Pincode[0] if len(Pincode) > 0 else "",
            "EmailID": EmailID[0] if len(EmailID) > 0 else "",
            "GuardianName": GuardianName[0] if len(GuardianName) > 0 else "",
            "CoachName": CoachName[0] if len(CoachName) > 0 else "",
            "registrationDate": registrationDate[0] if len(registrationDate) > 0 else "",
        }
    elif selection=="GYMNASTICS":
        name = re.findall(r"Name:\s*(.*?)\s*Address:", dd)
        Address = re.findall(r"Address:\s*(.*?)\s*Age:", dd)
        DateRegistion = [re.search(r"Date:(.*?)Date of Birth:", dd, re.DOTALL).group(1).strip()]
        Age = [re.search(r"Age:\s*([\d.]+)", dd).group(1).strip()]
        DOB = re.findall(r"Date of Birth:\s*(.*?)\s*Interests:", dd)
        Interests = re.findall(r"Interests:\s*(.*?)\s*Father's name and occupation", dd)
        Father_details = re.findall(r"Father's name and occupation:\s*(.*?)\s*Father's phone number, whatsapp and mail id:", dd)
        Father_contact = re.findall(r"Father's phone number, whatsapp and mail id:\s*(.*?)\s*Mother's name and occupation:", dd)
        Mother_details  = re.findall(r"Mother's name and occupation:\s*(.*?)\s*Mother's phone number, whatsapp and mail id:", dd)
        Mother_contact = [re.search(r"Mother's phone number, whatsapp and mail id:(.*?)Health problems/allergies, if any No", dd, re.DOTALL).group(1).strip()] # type: ignore
        healthIssues = re.findall(r"Health problems/allergies, if any No\s*(.*?)\s*Fees (to be paid before the first class)", dd)

        return {
            "name": name[0] if len(name) > 0 else "",
            "Address": Address[0] if len(Address) > 0 else "",
            "DateRegistion": DateRegistion[0] if len(DateRegistion) > 0 else "",
            "Age": Age[0] if len(Age) > 0 else "",
            "DOB": DOB[0] if len(DOB) > 0 else "",
            "Interests": Interests[0] if len(Interests) > 0 else "",
            "Father_details": Father_details[0] if len(Father_details) > 0 else "",
            "Father_contact": Father_contact[0] if len(Father_contact) > 0 else "",
            "Mother_details": Mother_details[0] if len(Mother_details) > 0 else "",
            "Mother_contact": Mother_contact[0] if len(Mother_contact) > 0 else "",
            "healthIssues": healthIssues[0] if len(healthIssues) > 0 else "",
        }
    else:
        return {}


def main():
    try:
        st.title("Multiple Images Collapsible Viewer")
        
        selection = st.sidebar.selectbox("Select an Form Format:", ["HORIZON_ACADEMY", "GYMNASTICS"], index=0)

        uploaded_files = st.sidebar.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write("### Uploaded Images:")

            for i, file in enumerate(uploaded_files):
                dd, markings = detect_handwritten_ocr(file.getvalue())
                
                image_url = upload_image_to_cloudinary(file)
                data_next = {
                    "url":image_url,
                    "markings":markings
                }
                image = Image.open(file)
                if selection:
                    data = parser_text(dd, selection)
                if image_url:
                    data['url'] = image_url
                else:
                    data['url'] = ""

                nparr = np.frombuffer(file.getvalue(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                person_img = extractPersonImg(img)
                ret, jpeg = cv2.imencode('.jpg', person_img)
                
                profile_pic_url = upload_image_to_cloudinary(jpeg.tobytes())
                
                if image_url:
                    data['profile_pic_url'] = profile_pic_url # type: ignore
                else:
                    data['profile_pic_url'] = ""
                
                allRes.update({file.name:data})
                allData.update({file.name:data_next})
                
                with st.expander(file.name, expanded=False):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(image, use_column_width=True)
                        if person_img is not None:
                            st.image(person_img, use_column_width=True)

                    with col2:
                        for key, value in data.items(): # type: ignore
                            new_val = st.text_input(key, key=f"{i}{key}", value=value)
                            if new_val != value:
                                data[key] = [new_val]
                        allRes[file.name] = data

        if st.button("Export to CSV"):
            for key, value in allData.items():
                bdy = {
                    "name":key,
                    "markings":value['markings'],
                    "url":value['url'],
                    "timestamp":time.time()
                }                    

                response = requests.post(st.secrets["URL"], json=bdy)
                if response.status_code == 200:
                    st.success("Data saved successfully!")
                else:
                    st.error(f"Failed to save data. Status code: {response.text}")
            df = pd.DataFrame.from_dict(allRes, orient="index")
            csv = df.to_csv(index_label="filename")
            b64 = base64.b64encode(csv.encode()).decode()
            href = f"data:file/csv;base64,{b64}"
            st.markdown(
                f'<a href="{href}" download="output.csv">Download CSV file</a>',
                unsafe_allow_html=True,
            )

    except Exception as e:
        traceback.print_exc()
        print(e)


if __name__ == "__main__":
    main()
