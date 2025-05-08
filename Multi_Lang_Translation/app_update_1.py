import streamlit as st
import pandas as pd
import io
import os
import re
import logging
from mtranslate import translate
from gtts import gTTS
import speech_recognition as sr
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY = 10
MAX_FILE_SIZE_MB = 5
DATA_PATH = os.getenv("LANGUAGE_CSV_PATH", "Multi_Lang_Translation/language.csv")
SUPPORTED_AUDIO_FORMATS = {"wav": "audio/wav", "mp3": "audio/mp3"}
SUPPORTED_TTS_LANGS = {
    "af", "ar", "bn", "en", "fr", "de", "hi", "gu", "it", "ja", "kn", "ml",
    "mr", "ta", "te", "ur", "zh-CN"
}
WORDCLOUD_CONFIG = {
    "width": 400,
    "height": 300,
    "background_color": "white",
    "min_font_size": 10,
    "colormap": "viridis"
}

@st.cache_data
def load_language_data():
    """Load and validate language dataset from CSV."""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Language dataset not found at '{DATA_PATH}'")
        df = pd.read_csv(DATA_PATH)
        df.dropna(inplace=True)
        lang_array = dict(zip(df['name'], df['iso']))
        return lang_array, list(lang_array.keys())
    except FileNotFoundError as e:
        logger.error(str(e))
        st.error(str(e))
        return {}, []
    except Exception as e:
        logger.error(f"Error loading language data: {str(e)}")
        st.error(f"Error loading language data: {str(e)}")
        return {}, []

def validate_input(text):
    """Validate and sanitize input text."""
    if not text or not text.strip():
        return None, "Input text cannot be empty."
    # Remove potentially harmful characters (basic sanitization)
    text = re.sub(r'[<>]', '', text.strip())
    if len(text) > 1000:  # Arbitrary limit to prevent abuse
        return None, "Input text is too long (max 1000 characters)."
    return text, None

def transcribe_audio(audio_file):
    """Transcribe audio file to text using Google Speech Recognition."""
    if not audio_file:
        return None, "No audio file provided."
    
    if audio_file.type not in SUPPORTED_AUDIO_FORMATS.values():
        return None, f"Unsupported audio format. Please upload {', '.join(SUPPORTED_AUDIO_FORMATS.keys())}."
    
    if audio_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return None, f"Audio file too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
    
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            logger.info(f"Successfully transcribed audio to: {text[:50]}...")
            return text, None
    except sr.UnknownValueError:
        logger.warning("Audio transcription failed: Could not understand audio")
        return None, "Could not understand the audio. Please try a clearer recording."
    except sr.RequestError as e:
        logger.error(f"Transcription service error: {str(e)}")
        return None, f"Transcription service error: {str(e)}. Please check your internet connection."
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {str(e)}")
        return None, f"Unexpected error during transcription: {str(e)}"

def generate_wordcloud(text):
    """Generate a word cloud from the input text."""
    try:
        if not text or not text.strip():
            return None, "No text provided for word cloud."
        wordcloud = WordCloud(**WORDCLOUD_CONFIG).generate(text)
        plt.figure(figsize=(4, 3))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        logger.info("Word cloud generated successfully")
        return buffer, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

def translate_and_generate_audio(text, lang_code):
    """Translate text and generate audio if supported."""
    if lang_code not in SUPPORTED_TTS_LANGS and lang_code not in [v for v in st.session_state.lang_array.values()]:
        return None, None, f"Unsupported language code: {lang_code}"
    
    try:
        translated = translate(text, lang_code)
        audio_buffer = None
        if lang_code in SUPPORTED_TTS_LANGS:
            tts = gTTS(text=translated, lang=lang_code, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            logger.info(f"Generated audio for language: {lang_code}")
        logger.info(f"Translated text to {lang_code}: {translated[:50]}...")
        return translated, audio_buffer, None
    except Exception as e:
        logger.error(f"Translation/audio generation error: {str(e)}")
        return None, None, f"Translation error: {str(e)}"

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "history": [],
        "selected_history_index": None,
        "displayed_input": "",
        "displayed_translated": "",
        "displayed_audio": None,
        "displayed_wordcloud": None,
        "is_history_view": False,
        "lang_array": {},
        "lang_names": [],
        "last_target_lang": None  # Track last used target language
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_output_section():
    """Render the translated text, audio, and word cloud."""
    if not st.session_state.displayed_translated:
        st.info("No translation available. Please provide text or audio input.")
        return
    
    st.subheader("🔁 Translated Text")
    if st.session_state.is_history_view:
        st.info("Showing translation from history")
    
    col_trans, col_cloud = st.columns([3, 2])
    with col_trans:
        st.write(st.session_state.displayed_translated)
        if st.session_state.displayed_audio:
            st.audio(st.session_state.displayed_audio, format='audio/mp3')
            st.download_button(
                label="Download Audio",
                data=st.session_state.displayed_audio,
                file_name="translated.mp3",
                mime="audio/mp3",
                key=f"download_audio_{uuid.uuid4()}"
            )
    with col_cloud:
        if st.session_state.displayed_wordcloud:
            st.image(
                st.session_state.displayed_wordcloud,
                caption="Word Cloud",
                use_column_width=True,
                output_format='PNG',
                clamp=True
            )
        else:
            st.write("No word cloud generated.")

def render_sidebar():
    """Render the sidebar with translation history."""
    st.sidebar.header("📜 Translation History")
    if not st.session_state.history:
        st.sidebar.info("No translation history yet.")
        return
    
    for i, (original, translated) in enumerate(reversed(st.session_state.history)):
        preview = " ".join(original.strip().split()[:3]) + ("..." if len(original.strip().split()) > 3 else "")
        translated_preview = translated[:20] + ("..." if len(translated) > 20 else "")
        if st.sidebar.button(
            f"{preview} -> {translated_preview}",
            key=f"history_{i}",
            help=f"View translation: {original[:50]}..."
        ):
            st.session_state.selected_history_index = len(st.session_state.history) - 1 - i
            original, translated = st.session_state.history[st.session_state.selected_history_index]
            st.session_state.displayed_input = original
            st.session_state.displayed_translated = translated
            st.session_state.is_history_view = True
            lang_code = st.session_state.lang_array[st.session_state.target_lang]
            if lang_code in SUPPORTED_TTS_LANGS:
                tts = gTTS(text=translated, lang=lang_code, slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                st.session_state.displayed_audio = audio_buffer
            else:
                st.session_state.displayed_audio = None
            st.session_state.displayed_wordcloud, _ = generate_wordcloud(original)
    
    if st.sidebar.button("Clear History", key="clear_history"):
        st.session_state.history = []
        st.session_state.selected_history_index = None
        st.session_state.is_history_view = False
        st.session_state.displayed_input = ""
        st.session_state.displayed_translated = ""
        st.session_state.displayed_audio = None
        st.session_state.displayed_wordcloud = None

def main():
    """Main Streamlit app function."""
    # Initialize session state
    initialize_session_state()

    # Page setup
    st.set_page_config(page_title="Translator", layout="wide", page_icon="🌍")
    st.title("🌍 Multilingual Translator with Audio & Word Cloud Visualization")
    st.markdown("Translate text or audio into multiple languages with word cloud visualization.")

    # Load language data
    with st.spinner("Loading language data..."):
        if not st.session_state.lang_array:
            st.session_state.lang_array, st.session_state.lang_names = load_language_data()
    if not st.session_state.lang_array:
        return

    # Layout: Language selector + Audio uploader
    col1, col2 = st.columns([2, 3])
    with col1:
        st.selectbox(
            "🎯 Select Target Language",
            st.session_state.lang_names,
            key="target_lang",
            help="Choose the language to translate into"
        )
    with col2:
        audio_file = st.file_uploader(
            "🎤 Upload Audio File",
            type=list(SUPPORTED_AUDIO_FORMATS.keys()),
            key="audio_uploader",
            help=f"Upload a {', '.join(SUPPORTED_AUDIO_FORMATS.keys())} file (max {MAX_FILE_SIZE_MB}MB)"
        )

    # Input handling
    user_input = None
    if not st.session_state.is_history_view:
        user_input = st.chat_input("Type a message to translate...", key="chat_input")
    else:
        st.text_input(
            "Original Text (from history)",
            value=st.session_state.displayed_input,
            disabled=True,
            key="history_input"
        )
        if st.button("New Translation", key="new_translation"):
            st.session_state.is_history_view = False
            st.session_state.selected_history_index = None
            st.session_state.displayed_input = ""
            st.session_state.displayed_translated = ""
            st.session_state.displayed_audio = None
            st.session_state.displayed_wordcloud = None

    # Reset button
    if st.button("Reset Input", key="reset_input"):
        st.session_state.displayed_input = ""
        st.session_state.displayed_translated = ""
        st.session_state.displayed_audio = None
        st.session_state.displayed_wordcloud = None
        st.session_state.is_history_view = False

    # Handle target language change
    if (st.session_state.last_target_lang != st.session_state.target_lang and 
        st.session_state.displayed_input and not st.session_state.is_history_view):
        with st.spinner("🔄 Re-translating for new language..."):
            validated_input, error = validate_input(st.session_state.displayed_input)
            if validated_input:
                try:
                    lang_code = st.session_state.lang_array[st.session_state.target_lang]
                except KeyError:
                    st.error(f"Invalid target language: {st.session_state.target_lang}")
                    return
                translated, audio_buffer, error = translate_and_generate_audio(validated_input, lang_code)
                if translated:
                    st.session_state.displayed_translated = translated
                    st.session_state.displayed_audio = audio_buffer
                    st.session_state.displayed_wordcloud, _ = generate_wordcloud(validated_input)
                    # Update history with new translation
                    st.session_state.history.append((validated_input, translated))
                    if len(st.session_state.history) > MAX_HISTORY:
                        st.session_state.history.pop(0)
                else:
                    st.error(error)
            else:
                st.error(error)
        st.session_state.last_target_lang = st.session_state.target_lang

    # Handle audio transcription
    if not st.session_state.is_history_view and audio_file and not user_input:
        with st.spinner("🔊 Transcribing audio..."):
            text, error = transcribe_audio(audio_file)
            if text:
                user_input = text
                st.success(f"Transcribed : {user_input}")
            else:
                st.error(error)

    # Translation logic for new input
    if not st.session_state.is_history_view and user_input:
        with st.spinner("🔄 Translating..."):
            validated_input, error = validate_input(user_input)
            if validated_input:
                try:
                    lang_code = st.session_state.lang_array[st.session_state.target_lang]
                except KeyError:
                    st.error(f"Invalid target language: {st.session_state.target_lang}")
                    return
                translated, audio_buffer, error = translate_and_generate_audio(validated_input, lang_code)
                if translated:
                    st.session_state.displayed_input = validated_input
                    st.session_state.displayed_translated = translated
                    st.session_state.displayed_audio = audio_buffer
                    st.session_state.displayed_wordcloud, _ = generate_wordcloud(validated_input)
                    st.session_state.history.append((validated_input, translated))
                    if len(st.session_state.history) > MAX_HISTORY:
                        st.session_state.history.pop(0)
                    st.session_state.last_target_lang = st.session_state.target_lang
                else:
                    st.error(error)
            else:
                st.error(error)

    # Render output
    render_output_section()
    
    # Render sidebar
    render_sidebar()

if __name__ == "__main__":
    main()
