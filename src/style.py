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


# footer
def add_footer():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600&display=swap');
            
            .footer-container {
                font-family: 'Raleway', sans-serif;
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                padding: 10px 0;
                text-align: center;
                z-index: 1000;
            }
            
            .footer-content {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .footer-text {
                font-size: 16px;
                font-weight: 800;
                margin: 0;
                padding: 0 20px;
            }
            
            .footer-link {
                font-weight: 600;
                text-decoration: none;
                position: relative;
                transition: all 0.3s ease;
                padding: 5px 10px;
                border-radius: 5px;
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            
            /* Theme-adaptive styles */
            @media (prefers-color-scheme: light) {
                :root {
                    # --text-color: #262730;
                    --link-color: #075E54;
                    --link-hover-bg: rgba(7, 94, 84, 0.1);
                    --link-hover-shadow: rgba(7, 94, 84, 0.2);
                }
            }
            
            @media (prefers-color-scheme: dark) {
                :root {
                    # --background-color: #0e1117;
                    # --text-color: #fafafa;
                    --link-color: #25D366;
                    --link-hover-bg: rgba(37, 211, 102, 0.1);
                    --link-hover-shadow: rgba(37, 211, 102, 0.2);
                }
            }
            
            /* Streamlit-specific theme detection */
            [data-testid="stAppViewContainer"] {
                color: var(--text-color);
            }
        </style>
        <div class="footer-container">
            <div class="footer-content">
                <p class="footer-text">
                    Developed by
                    <a href="https://github.com/Abdulraqib20" target="_blank" class="footer-link">raqibcodes</a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
