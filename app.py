import streamlit as st
import cv2
import numpy as np
from core.ocr_engine import OCREngine
from core.chat_parser import ChatParser
from core.llm_client import LLMClient
from core.i18n import t
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Daisy - Your Dating Wingman", page_icon="💘", layout="wide")

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'

# Custom CSS for chat bubbles
st.markdown("""
<style>
.chat-bubble {
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 70%;
    word-wrap: break-word;
}
.me-bubble {
    background-color: #DCF8C6;
    color: black;
    margin-left: auto;
    text-align: right;
    border-bottom-right-radius: 0;
}
.target-bubble {
    background-color: #FFFFFF;
    color: black;
    margin-right: auto;
    text-align: left;
    border-bottom-left-radius: 0;
    border: 1px solid #E0E0E0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ocr_engine():
    return OCREngine()

def main():
    st.title(t("main_title"))
    st.markdown(t("subtitle"))

    # Sidebar for configuration
    with st.sidebar:
        st.header(t("settings"))
        
        # Language Selection
        lang_map = {"English": "en", "中文": "zh"}
        # Find index of current language
        current_lang_code = st.session_state.get('language', 'en')
        current_index = list(lang_map.values()).index(current_lang_code)
        
        selected_lang = st.selectbox(
            "Language / 语言", 
            options=list(lang_map.keys()), 
            index=current_index
        )
        
        # Update session state if changed
        new_lang_code = lang_map[selected_lang]
        if new_lang_code != current_lang_code:
            st.session_state['language'] = new_lang_code
            st.rerun()

        st.divider()
        
        # Model Selection
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "deepseek-chat", "deepseek-reasoner"]
        model = st.selectbox(t("model_label"), model_options, index=0)
        
        # Base URL Logic
        default_base_url = "https://api.openai.com/v1"
        if "deepseek" in model:
            default_base_url = "https://api.deepseek.com"
            
        base_url = st.text_input(t("base_url_label"), value=os.getenv("OPENAI_BASE_URL", default_base_url))
        
        # API Key
        api_key_label = "DeepSeek API Key" if "deepseek" in model else t("api_key_label")
        api_key = st.text_input(api_key_label, value=os.getenv("OPENAI_API_KEY", ""), type="password")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        st.info(f"{t('ensure_api_key')}")

    # File uploader
    uploaded_file = st.file_uploader(t("upload_label"), type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Convert to numpy array for processing
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Display image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption=t("uploaded_caption"), use_container_width=True)
            
        with col2:
            st.subheader(t("analysis_result"))
            
            if st.button(t("analyze_btn")):
                with st.spinner(t("analyzing_spinner")):
                    try:
                        # Initialize engines
                        ocr_engine = get_ocr_engine()
                        parser = ChatParser()
                        
                        # OCR
                        ocr_results = ocr_engine.extract_text(image)
                        
                        # Parse
                        height, width, _ = image.shape
                        messages = parser.parse(ocr_results, width)
                        
                        if not messages:
                            st.warning(t("no_messages_warning"))
                        else:
                            st.session_state['chat_history'] = messages
                            st.success(t("detected_messages", count=len(messages)))
                            
                    except Exception as e:
                        st.error(t("error_analysis", error=str(e)))

            # Display parsed chat with editing capabilities
            if 'chat_history' in st.session_state:
                st.write(f"### {t('edit_chat_history')}")
                
                # Create a list to track indices to remove
                indices_to_remove = []
                
                for i, msg in enumerate(st.session_state['chat_history']):
                    col1, col2, col3 = st.columns([1, 4, 0.5])
                    
                    with col1:
                        # Role selector
                        role_options = ["me", "target"]
                        role_labels = [t("chat_role_me"), t("chat_role_target")]
                        current_index = 0 if msg['role'] == 'me' else 1
                        
                        new_role = st.selectbox(
                            f"Role {i}", 
                            role_options, 
                            index=current_index, 
                            format_func=lambda x: role_labels[role_options.index(x)],
                            key=f"role_{i}",
                            label_visibility="collapsed"
                        )
                        st.session_state['chat_history'][i]['role'] = new_role
                        
                    with col2:
                        # Text editor
                        new_text = st.text_input(
                            f"Text {i}", 
                            value=msg['text'], 
                            key=f"text_{i}",
                            label_visibility="collapsed"
                        )
                        st.session_state['chat_history'][i]['text'] = new_text
                        
                    with col3:
                        # Delete button
                        if st.button("🗑️", key=f"del_{i}", help=t("delete_message")):
                            indices_to_remove.append(i)
                
                # Remove deleted messages
                if indices_to_remove:
                    for index in sorted(indices_to_remove, reverse=True):
                        del st.session_state['chat_history'][index]
                    st.rerun()
                
                # Add new message button
                if st.button(t("add_message")):
                    st.session_state['chat_history'].append({'role': 'target', 'text': ''})
                    st.rerun()
                
                st.divider()
                
                # Additional Context Inputs
                col_stage, col_gender = st.columns(2)
                with col_stage:
                    stage_options = ["icebreaking", "ambiguous", "dating", "stable"]
                    stage_labels = [t("stage_icebreaking"), t("stage_ambiguous"), t("stage_dating"), t("stage_stable")]
                    selected_stage = st.selectbox(t("current_stage_label"), stage_options, format_func=lambda x: stage_labels[stage_options.index(x)])
                    
                with col_gender:
                    gender_options = ["male", "female", "unknown"]
                    gender_labels = [t("gender_male"), t("gender_female"), t("gender_unknown")]
                    target_gender = st.selectbox(t("target_gender_label"), gender_options, format_func=lambda x: gender_labels[gender_options.index(x)])
                
                # Reply Generation
                if st.button(t("generate_reply_btn")):
                    if not api_key:
                        st.error(t("enter_api_key_error"))
                    else:
                        # Update prompt context
                        llm_client = LLMClient(api_key=api_key, model=model, base_url=base_url)
                        
                        # Add context to history text for LLM
                        context_note = f"\n[Context Info] Relationship Stage: {selected_stage}, Target Gender: {target_gender}"
                        # We append this context note temporarily to the last message or handle it in client
                        # For now, let's append it to the chat history passed to LLM (non-destructively)
                        
                        chat_history_with_context = st.session_state['chat_history'].copy()
                        chat_history_with_context.append({'role': 'system', 'text': context_note})
                        
                        with st.spinner(t("consulting_coach")):
                            result = llm_client.generate_reply(chat_history_with_context)
                            
                            # Handle error
                            if result.get("error"):
                                st.error(t("error_llm", error=result["error"]))
                            else:
                                # 1. Display Reasoning (if available, e.g. for DeepSeek-R1)
                                if result.get("reasoning"):
                                    with st.expander("🧠 DeepSeek-R1 Reasoning (Thinking Process)", expanded=False):
                                        st.markdown(result["reasoning"])
                                
                                # 2. Display Prompts (Debug info for tuning)
                                with st.expander("🛠️ Debug: View Prompts", expanded=False):
                                    st.text("System Prompt:")
                                    st.code(result["system_prompt"])
                                    st.text("User Prompt:")
                                    st.code(result["user_prompt"])

                                # 3. Display Final Reply
                                st.write(f"### {t('suggested_replies')}")
                                st.markdown(result["content"])

if __name__ == "__main__":
    main()
