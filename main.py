from core_1 import run_llm

import streamlit as st
from streamlit_chat import message

import streamlit as st

from PIL import Image

image = Image.open("banner_checker_1.png")
st.image(image)

st.header("EDW EASA Regul. Implementation checker")

# if "user_promt_history" not in st.session_state:
#     st.session_state["user_promt_history"] = []
# if "chat_answer_history" not in st.session_state:
#     st.session_state["chat_answer_history"] = []
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
# if "follow_up_count" not in st.session_state:
#     st.session_state["follow_up_count"] = 0

with st.form("promt_input", clear_on_submit=True):
    prompt = st.text_area("Prompt:", placeholder="Enter EASA regulation text here...")
    # if st.session_state["follow_up_count"] < 2:
    #     submitted = st.form_submit_button("Follow up")
    # else:
    #     submitted = st.form_submit_button("Follow up", disabled=True)
    submitted_new = st.form_submit_button("New regulation")

    # if submitted and prompt:
    #     with st.spinner("generating response..."):
    #         generated_response = run_llm(
    #             query=prompt, chat_history=st.session_state["chat_history"]
    #         )

    #         formatted_response = f"{generated_response}"
    #         st.session_state["user_promt_history"].append(prompt)
    #         st.session_state["chat_answer_history"].append(formatted_response)
    #         st.session_state["chat_history"].append((prompt, generated_response))
    #         st.session_state["follow_up_count"] += 1

    if submitted_new and prompt:
        # st.session_state["chat_history"] = []
        # st.session_state["user_promt_history"] = []
        # st.session_state["chat_answer_history"] = []
        # st.session_state["follow_up_count"] = 0

        with st.spinner("Generating response.."):
            generated_response = run_llm(query=prompt)

            # formatted_response = f"{generated_response}"

            # st.session_state["user_promt_history"].append(prompt)
            # st.session_state["chat_answer_history"].append(formatted_response)
            # st.session_state["chat_history"].append((prompt, generated_response))

        # if st.session_state["chat_answer_history"]:
        #     for generated_response, user_query in zip(
        #         st.session_state["chat_answer_history"], st.session_state["user_promt_history"]
        #     ):
        message(prompt, is_user=True)
        message(generated_response)
