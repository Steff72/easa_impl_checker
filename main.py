from core import run_llm

import streamlit as st
from streamlit_chat import message

import streamlit as st

from PIL import Image

image = Image.open("banner_checker_1.png")
st.image(image)

st.header("EDW EASA Regul. Implementation checker")


with st.form("promt_input", clear_on_submit=True):
    prompt = st.text_area("Prompt:", placeholder="Enter EASA regulation text here...")

    submitted_new = st.form_submit_button("New regulation")

    if submitted_new and prompt:
        with st.spinner("Generating response.."):
            generated_response = run_llm(query=prompt)

        message(prompt, is_user=True)
        message(generated_response)
