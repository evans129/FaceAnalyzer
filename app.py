import streamlit as st
import cv2
import os
from PIL import Image
import  numpy as np
import tensorflow as tf
from keras.models import load_model
import math
import yagmail

caspath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(caspath)
fc=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt.xml"
lc=os.path.dirname(cv2.__file__)+"/data/haarcascade_lefteye_2splits.xml"
rc=os.path.dirname(cv2.__file__)+"/data/haarcascade_righteye_2splits.xml"
face = cv2.CascadeClassifier(fc)
leye = cv2.CascadeClassifier(lc)
reye = cv2.CascadeClassifier(rc)
lbl = ['Close', 'Open']
path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
rpred = [99]
lpred = [99]
color = (0, 255, 0)
model = load_model('models/cnnCat2.h5')
new_model=tf.keras.models.load_model("models/trained_mdl.h5")
new_model1=tf.keras.models.load_model("models/trained_model.h5")
#new_model=cv2.face.LBPHFaceRecognizer_create()
def detect_faces(image,ans,emo,ans1):
    img1=image
    img2=image
    FONT_SCALE = 2e-3  # Adjust for larger font size in all images
    THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

    height, width= image.size

    font_scale = min(width, height) * FONT_SCALE
    thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list = []
    preds = []
    for (x, y, w, h) in faces:
     roi_gray = gray[y:y + h, x:x + w]
     roi_color = img[y:y + h, x:x + w]
     facess = face_cascade.detectMultiScale(roi_gray)
     if (len(facess) == 0):
        st.write("Face not detected")
        print("face not detected")
     else:
        for (ex, ey, ew, eh) in facess:
            face_frame = img[y:y + h, x:x + w]
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = tf.keras.preprocessing.image.img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame = tf.keras.applications.mobilenet_v2.preprocess_input(face_frame)
            faces_list.append(face_frame)
            if len(faces_list) > 0:
                preds = new_model.predict(faces_list)
            for pred in preds:
                (mask, withoutMask) = pred
            if mask > withoutMask:
                label = "Mask"
            else:
                label = "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255,0,0)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            for pred in preds:
                # mask contain probabily of wearing a mask and vice versa
                (mask, withoutMask) = pred
                print(mask, withoutMask)
            if mask > withoutMask:
                st.write('Wearing a Mask')
                print('Wearing Mask')
                ans= 'Wearing a Mask'
            else:
                st.write('Not Wearing a Mask')
                print('Not wearing Mask')
                ans='Not Wearing a Mask'
    img1 = np.array(img1.convert('RGB'))
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        facess = face_cascade.detectMultiScale(roi_gray)
        if (len(facess) == 0):
            st.write("Face not detected")
            print("face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey:ey + eh, ex:ex + ew]
                final_image = cv2.resize(face_roi, (224, 224))
                final_image = np.expand_dims(final_image, axis=0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                Predictions = new_model1.predict(final_image)
                font_scale = 1.5
                if (np.argmax(Predictions) == 0):
                    emo = "ANGRY"

                elif (np.argmax(Predictions) == 1):
                    emo = "DISGUST"

                if (np.argmax(Predictions) == 2):
                    emo = "FEAR"

                if (np.argmax(Predictions) == 3):
                    emo = "HAPPY"

                if (np.argmax(Predictions) == 4):
                    emo = "NEUTRAL"

                if (np.argmax(Predictions) == 5):
                    emo = "SAD"

                elif (np.argmax(Predictions) == 6):
                    emo = "SURPRISE"
    st.write("Face looks " +emo)
    img2 = np.array(img2.convert('RGB'))
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
    for (x, y, w, h) in right_eye:
        r_eye = img[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        predict_r = model.predict(r_eye)
        rpred = np.argmax(predict_r, axis=1)
        # rpred = model.predict_classes(r_eye)
        if (rpred[0] == 1):
            lbl = 'Open'
        if (rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = img[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        predict_l = model.predict(l_eye)
        lpred = np.argmax(predict_l, axis=1)
        # lpred = model.predict_classes(l_eye)
        if (lpred[0] == 1):
            lbl = 'Open'
        if (lpred[0] == 0):
            lbl = 'Closed'
        break

    if (rpred[0] == 0 and lpred[0] == 0):
        ans1 += 'Closed'
        st.write("Eyes are " +ans1)


    else:
        ans1+= 'Open'
        st.write("Eyes are "+ans1)



    return img,ans,emo,ans1
def main():
    ans=''
    dat=''
    emo = ''
    ans1=' '
    st.title("Face Analyzer")
    html_temp="""
    <body style="background-color:red;">
    <div style="background-color:#ff6f61;padding:10px">
    <h2 style="color:white;text-align:center;">Face Analysis app</h2>
    </div>
    </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.text("original Image")
        st.image(image)
    else:
     image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
     if image_file is not None:
        image=Image.open(image_file)
        st.text("original Image")
        st.image(image)
    if st.button("Recognise"):
        result_img,ans,emotion,ans1=detect_faces(image,ans,emo,ans1)
        st.image(result_img)
        dat="The Uploaded Face is "+ ans +" , "+ "looks "+emotion+" and the Eyes are "+ans1
        st.download_button("Download Analysis", dat, file_name='Processed_Image.txt', key='Download Image Analysis')
    rec=st.text_input('Enter your email')
    if st.button("Mail to me"):
          result_img, ans,emotion,ans1 = detect_faces(image, ans,emo,ans1)
          dat = "The Uploaded Face is "+ ans +" , "+ "looks "+emotion+" and the eyes are "+ans1
          if(len(rec)!=0):
           yag = yagmail.SMTP('devanshkumaravi@gmail.com', 'oowhmqyyreotkwys')
           contents = [dat]
           yag.send(rec, 'subject', contents)
          else:
           st.warning('Please Enter Your Email', icon="⚠️")

if __name__== '__main__':
    main()