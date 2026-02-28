import streamlit as st
import cv2
import numpy as np
from core.ocr_engine import OCREngine
from core.chat_parser import ChatParser
from core.llm_client import LLMClient
from core.i18n import t
from core.logger import get_logger
from core.image_processor import ImageProcessor
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
    border-radius: 20px;
    margin-bottom: 10px;
    max-width: 70%;
    word-wrap: break-word;
    font-size: 16px;
}
.target-bubble {
    background-color: #f0f0f0;
    color: #000;
    border-bottom-left-radius: 5px;
}
.me-bubble {
    background-color: #0084ff;
    color: #fff;
    border-bottom-right-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ocr_engine():
    return OCREngine()

def display_logs():
    """Display logs in the sidebar for debugging."""
    if st.sidebar.checkbox("Show Logs (Debug)", value=False):
        st.sidebar.markdown("### System Logs")
        log_file_path = os.path.join("logs", "daisy.log")
        if os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8") as f:
                logs = f.readlines()
                # Show last 50 lines
                st.sidebar.code("".join(logs[-50:]), language="text")
        else:
            st.sidebar.warning("No logs found.")

def main():
    logger = get_logger("app")
    logger.info("Application started")
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
            logger.info(f"Language changed from {current_lang_code} to {new_lang_code}")
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

        # System Prompt Config
        with st.expander(t("system_prompt_config")):
            st.session_state['system_prompt'] = st.text_area(
                t("edit_prompt"), 
                value=st.session_state.get('system_prompt', LLMClient.get_default_prompt()),
                height=300
            )

    # Display Logs Component
    display_logs()

    # File uploader
    uploaded_files = st.file_uploader(t("upload_label"), type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if uploaded_files:
        # Convert to numpy array for processing
        raw_images = []
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            raw_images.append(cv2.imdecode(file_bytes, 1))
            logger.info(f"Image uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")

        # Process and Stitch
        with st.spinner(t("analyzing_spinner")): # Reusing spinner text or creating new key
            if len(raw_images) > 1:
                status_placeholder = st.empty()
                status_placeholder.info(t("stitching_images", count=len(raw_images)))
                image = ImageProcessor.process(raw_images, denoise=True)
                status_placeholder.success(t("stitching_success", count=len(raw_images)))
            else:
                image = raw_images[0]
                # Optional: Denoise single image too
                # image = ImageProcessor.denoise_image(image)
            
            # Save the processed image to session state so we don't re-process on rerun
            st.session_state['processed_image'] = image

        # Display image (Use processed image)
        if 'processed_image' in st.session_state:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(st.session_state['processed_image'], caption=t("uploaded_caption"), use_container_width=True)
                
            with col2:
                st.subheader(t("analysis_result"))
                
                if st.button(t("analyze_btn")):
                    with st.spinner(t("analyzing_spinner")):
                        try:
                            # Initialize engines
                            ocr_engine = get_ocr_engine()
                            parser = ChatParser()
                            
                            # OCR using the PROCESSED image
                            ocr_results = ocr_engine.extract_text(st.session_state['processed_image'])
                            
                            # Parse
                            height, width, _ = st.session_state['processed_image'].shape
                            messages = parser.parse(ocr_results, width)
                            
                            # LLM-based OCR Correction (Optional but recommended)
                            if messages and api_key:
                                with st.status(t("correcting_ocr"), expanded=False) as status:
                                    llm_client = LLMClient(api_key=api_key, model=model, base_url=base_url)
                                    messages = llm_client.correct_ocr_errors(messages)
                                    status.update(label="OCR Correction Complete!", state="complete", expanded=False)

                            if not messages:
                                    logger.warning("No messages detected in uploaded image")
                                    st.warning(t("no_messages_warning"))
                            else:
                                logger.info(f"Successfully extracted {len(messages)} messages")
                                st.session_state['chat_history'] = messages
                                st.success(t("detected_messages", count=len(messages)))
                            
                        except Exception as e:
                            logger.error(f"Analysis failed: {e}", exc_info=True)
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
                        
                        # Get current UI language
                        ui_lang = st.session_state.get('language', 'en')
                        
                        with st.spinner(t("consulting_coach")):
                            # Pass ui_language to generate_reply
                            result = llm_client.generate_reply(chat_history_with_context, ui_language=ui_lang)
                            
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
