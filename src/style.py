import streamlit as st

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
            border-radius: 12px;
            border: 2px solid #4CAF50;
        }
        .stButton > button:hover {
            background-color: white;
            color: black;
        }
        .stTextInput > div > div > input {
            border-radius: 12px;
        }
        .stSelectbox > div > div > select {
            border-radius: 12px;
        }
        .stSlider > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)
