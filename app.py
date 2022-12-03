import time

import streamlit as st
import os,tempfile
from pathlib import Path

from PIL import Image

st.set_page_config(page_title="Emergency Vehicle Detection", page_icon="./icon.jpg", layout="wide", initial_sidebar_state="auto", menu_items=None)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            button[data-baseweb="tab"]{font-size:22px;}
            </style>
            """
st.markdown(hide_streamlit_style,unsafe_allow_html=True)


st.title("Emergency Vehicle Detection")

st.markdown("----", unsafe_allow_html=True)

image = st.file_uploader("Upload image",type=['jpg','jpeg','png'])

columns = st.columns((4.3, 1, 4.3))

if columns[1].button('Run Model'):
    if (image == None):
        st.warning('Please Upload All Files', icon="⚠️")
    else:
        # time.sleep(5)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_1_file:
            # st.markdown("## Original video file")
            fp = Path(tmp_1_file.name)
            fp.write_bytes(image.getvalue())
            image_path = tmp_1_file.name
            os.system('python emergency.py '+image_path)
            time.sleep(1)
            im = Image.open(image_path)
            st.image(im)
            with open('emer.txt', 'r') as file:
                for line in file.readlines():
                    st.write(line)
