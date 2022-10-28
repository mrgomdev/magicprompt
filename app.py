"""
From https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion
"""
import streamlit as st

import transformers

@st.experimental_singleton
def get_pipe():
    gpt2_pipe = transformers.pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2', device=0)
    return gpt2_pipe
prompt = st.text_input('prompt')
if prompt is not None:
    out = get_pipe()(prompt)
    st.write(out)
