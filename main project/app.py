import csv
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
from classifier import load_classifier,classify
from imagecaption import load_captioner,generate_caption
DB=Path(__file__).resolve().parent/'database.csv'
COLUMNS=['timestamp_utc','input_type','user_text','image_caption','classification','confidence']
def save_record(record):
    first=not DB.exists() or DB.stat().st_size==0
    with DB.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS)
        if first: writer.writeheader()
        writer.writerow(record)
@st.cache_resource
def get_classifier():return load_classifier()
@st.cache_resource
def get_captioner():return load_captioner()
st.title('Cellula — Toxic Content Classifier')
st.caption('Research demo: classifications may be incorrect. Dataset labels require further audit.')
tab1,tab2,tab3=st.tabs(['Classify text','Classify image','View CSV database'])
with tab1:
    text=st.text_area('Enter text')
    if st.button('Classify text'):
        if not text.strip():st.warning('Enter text first.')
        else:
            try:
                label,confidence=classify(text,'',get_classifier())
                save_record(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),input_type='text',user_text=text,image_caption='',classification=label,confidence=round(confidence,4)))
                st.success(f'Classification: {label} (model confidence: {confidence:.1%})')
            except Exception as exc:st.error(f'Cannot classify: {exc}')
with tab2:
    uploaded=st.file_uploader('Upload image',type=['png','jpg','jpeg'])
    optional_text=st.text_input('Optional accompanying user text')
    if st.button('Caption and classify image'):
        if uploaded is None:st.warning('Upload an image first.')
        else:
            try:
                image=Image.open(uploaded).convert('RGB')
                caption=generate_caption(image,get_captioner())
                st.write('Generated caption:',caption)
                label,confidence=classify(optional_text,caption,get_classifier())
                save_record(dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),input_type='image',user_text=optional_text,image_caption=caption,classification=label,confidence=round(confidence,4)))
                st.success(f'Classification: {label} (model confidence: {confidence:.1%})')
            except Exception as exc:st.error(f'Cannot process image: {exc}')
with tab3:
    if st.button('Refresh database') or DB.exists():
        if DB.exists() and DB.stat().st_size:st.dataframe(pd.read_csv(DB),use_container_width=True)
        else:st.info('No submissions yet.')
