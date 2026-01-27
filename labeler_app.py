# Save this as labeler_app.py
import streamlit as st
import pandas as pd
import random
from datetime import datetime
from datasets import load_dataset
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# Page config
st.set_page_config(page_title="Twitch Sentiment Labeler", layout="centered")

# Initialize session state
if 'current_message' not in st.session_state:
    st.session_state.current_message = None
if 'message_index' not in st.session_state:
    st.session_state.message_index = None
if 'labeled_count' not in st.session_state:
    st.session_state.labeled_count = 0
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sheet_connected' not in st.session_state:
    st.session_state.sheet_connected = False
if 'sheet' not in st.session_state:
    st.session_state.sheet = None


# Load dataset once
@st.cache_resource
def load_twitch_data():
    """Load and cache the Twitch dataset from HuggingFace"""
    try:
        dataset = load_dataset("lparkourer10/twitch_chat")
        messages = [msg.get('message', msg.get('text', str(msg))) for msg in dataset['train']]
        return messages
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []


# Google Sheets functions
@st.cache_resource
def init_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        # Get credentials from Streamlit secrets
        creds_dict = st.secrets["google_sheets"]

        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        # Open the spreadsheet
        spreadsheet = client.open('Twitch_Sentiment_Labels')
        sheet = spreadsheet.sheet1

        return sheet, True
    except Exception as e:
        st.error(f"❌ Google Sheets connection failed: {e}")
        return None, False


def load_labels_from_sheet(sheet):
    """Load all labels from Google Sheet"""
    try:
        records = sheet.get_all_records()
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return pd.DataFrame()


def save_label_to_sheet(sheet, message_id, message, sentiment, confidence, labeler_name, timestamp):
    """Save a single label to Google Sheet"""
    try:
        sheet.append_row([
            message_id,
            message,
            sentiment,
            confidence,
            labeler_name,
            timestamp
        ])
        return True
    except Exception as e:
        st.error(f"Error saving to sheet: {e}")
        return False


# Title and description
st.title("🎮 Twitch Chat Sentiment Labeler")
st.markdown("Label Twitch chat messages by sentiment. Help train our ML model!")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    labeler_name = st.text_input("Your Name:", placeholder="e.g., Student 1")

    st.divider()

    # Google Sheets Connection
    st.subheader("📊 Google Sheets Setup")

    if st.button("🔗 Connect to Google Sheets", use_container_width=True):
        with st.spinner("Connecting to Google Sheets..."):
            sheet, connected = init_google_sheets()
            if connected:
                st.session_state.sheet = sheet
                st.session_state.sheet_connected = True
                st.success("✅ Connected to Google Sheets!")
            else:
                st.session_state.sheet_connected = False

    if st.session_state.sheet_connected:
        st.success("✅ Google Sheets connected")

        # Show current stats from sheet
        if st.button("📂 Refresh Stats from Sheet", use_container_width=True):
            with st.spinner("Loading labels..."):
                df = load_labels_from_sheet(st.session_state.sheet)
                if not df.empty:
                    df_user = df[df['labeled_by'] == labeler_name] if labeler_name else df
                    st.metric("Your Labels", len(df_user))
                    st.metric("Total Labels", len(df))
    else:
        st.warning("⚠️ Not connected to Google Sheets yet")

    st.divider()

    # Load dataset
    if st.button("📥 Load Twitch Dataset", use_container_width=True):
        with st.spinner("Loading Twitch dataset..."):
            messages = load_twitch_data()
            if messages:
                st.session_state.messages = messages
                st.session_state.dataset_loaded = True
                st.success(f"✅ Loaded {len(messages)} messages!")

    if st.session_state.dataset_loaded:
        st.info(f"📊 Dataset ready: {len(st.session_state.messages)} messages available")

    st.divider()
    st.subheader("📖 Sentiment Guide")

    with st.expander("😄 Excitement"):
        st.write("""
        **Positive emotion about gameplay**
        - Examples: POGGERS, LETS GO, CLUTCH, HOLY
        - Signs: Caps lock, !, positive gaming emotes
        """)

    with st.expander("😠 Frustration"):
        st.write("""
        **Negative emotion, disappointment**
        - Examples: wtf, trash, ff15, throw
        - Signs: Curse words, negative words, ?
        """)

    with st.expander("😂 Humor"):
        st.write("""
        **Jokes, memes, sarcasm, playfulness**
        - Examples: KEKW, copypasta, laugh emotes
        - Signs: Laugh emotes, joke structure, sarcasm
        """)

    with st.expander("❓ Confusion"):
        st.write("""
        **Questions, not understanding**
        - Examples: what happened?, ???, how?
        - Signs: Question marks, confusion emotes
        """)

    with st.expander("😴 Boredom"):
        st.write("""
        **Lack of interest, slow pace**
        - Examples: ResidentSleeper, zzzz, boring
        - Signs: Sleep emotes, pace complaints
        """)

    with st.expander("😐 Neutral"):
        st.write("""
        **Everything else**
        - Examples: hi, gg, nice, general chat
        - Signs: Informational, greetings, generic
        """)

# Main content
if not st.session_state.dataset_loaded:
    st.warning("⚠️ Click '📥 Load Twitch Dataset' in the sidebar to begin!")
elif not st.session_state.sheet_connected:
    st.warning("⚠️ Click '🔗 Connect to Google Sheets' in the sidebar to sync labels!")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔄 Load Random Message", use_container_width=True):
            if st.session_state.messages:
                st.session_state.current_message = random.choice(st.session_state.messages)
                st.session_state.message_index = random.randint(10000, 99999)

    with col2:
        if st.button("⏭️ Skip Message", use_container_width=True):
            st.session_state.current_message = None
            st.session_state.message_index = None

    # Display current message
    if st.session_state.current_message:
        st.divider()

        st.markdown("### 💬 Current Message")
        message_container = st.container(border=True)
        with message_container:
            st.write(f"**ID:** {st.session_state.message_index}")
            st.markdown(f'```\n{st.session_state.current_message}\n```')

        st.divider()

        st.markdown("### 🏷️ Select Sentiment")

        col1, col2 = st.columns(2)

        with col1:
            sentiment = st.selectbox(
                "Sentiment:",
                ["Select...", "Excitement", "Frustration", "Humor", "Confusion", "Boredom", "Neutral"],
                index=0,
                key="sentiment_select"
            )

        with col2:
            confidence = st.selectbox(
                "Confidence:",
                ["Select...", "1 - Very Unsure", "2 - Unsure", "3 - Neutral", "4 - Confident", "5 - Very Confident"],
                index=0,
                key="confidence_select"
            )

        st.divider()

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("✅ Submit Label", use_container_width=True, type="primary"):
                if sentiment != "Select..." and confidence != "Select...":
                    confidence_score = int(confidence[0])
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Save to Google Sheet
                    if save_label_to_sheet(
                            st.session_state.sheet,
                            st.session_state.message_index,
                            st.session_state.current_message,
                            sentiment,
                            confidence_score,
                            labeler_name,
                            timestamp
                    ):
                        st.success(
                            f"✅ Labeled as **{sentiment}** (Confidence: {confidence_score}/5) and saved to Google Sheets!")
                        st.session_state.labeled_count += 1
                        st.balloons()

                        st.session_state.current_message = None
                        st.session_state.message_index = None
                    else:
                        st.error("❌ Failed to save to Google Sheets")

                else:
                    st.error("⚠️ Please select both sentiment and confidence!")
    else:
        if st.session_state.dataset_loaded and st.session_state.sheet_connected:
            st.info("👈 Click 'Load Random Message' to start labeling!")

    # Progress tracker
    st.divider()
    st.markdown("### 📈 Labeling Progress (This Session)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messages Labeled", st.session_state.labeled_count, "this session")
    with col2:
        st.metric("Target", 500, "for your team member")
    with col3:
        progress_pct = min((st.session_state.labeled_count / 500) * 100, 100)
        st.metric("Progress", f"{progress_pct:.1f}%")

    progress_bar = st.progress(min(st.session_state.labeled_count / 500, 1.0))

    # Show all labeled data from sheet
    st.divider()
    st.markdown("### 💾 All Labels (From Google Sheets)")

    if st.button("🔄 Refresh All Labels", use_container_width=True):
        with st.spinner("Loading all labels from Google Sheets..."):
            df_all = load_labels_from_sheet(st.session_state.sheet)

            if not df_all.empty:
                st.success(f"✅ Loaded {len(df_all)} total labels")

                # Overall stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Labels", len(df_all))
                with col2:
                    if labeler_name:
                        df_user = df_all[df_all['labeled_by'] == labeler_name]
                        st.metric(f"Your Labels", len(df_user))
                with col3:
                    st.metric("Team Members", df_all['labeled_by'].nunique())

                # Sentiment distribution
                st.subheader("Sentiment Distribution (All)")
                sentiment_counts = df_all['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

                # By labeler
                st.subheader("Labels by Team Member")
                labeler_counts = df_all['labeled_by'].value_counts()
                st.bar_chart(labeler_counts)

                # Recent labels
                st.subheader("Recent Labels (Latest 15)")
                st.dataframe(df_all.tail(15).iloc[::-1], use_container_width=True)
            else:
                st.info("No labels yet. Start labeling!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Twitch Sentiment Labeling Tool | CS 175 Project<br>
Labels synced to Google Sheets
</div>
""", unsafe_allow_html=True)