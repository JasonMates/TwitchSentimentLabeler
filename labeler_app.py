import streamlit as st
import pandas as pd
import random
from datetime import datetime
from datasets import load_dataset
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(page_title="Twitch Sentiment Labeler", layout="centered")

# initialize session
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


# load dataset once
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


# google Sheets functions
@st.cache_resource
def init_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        # get credentials from Streamlit secrets
        creds_dict = st.secrets["google_sheets"]

        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)

        # open the spreadsheet
        spreadsheet = client.open('Twitch_Sentiment_Labels')
        sheet = spreadsheet.sheet1

        return sheet, True
    except Exception as e:
        st.error(f"❌ Google Sheets connection failed: {e}")
        return None, False


def load_labels_from_sheet(sheet):
    try:
        records = sheet.get_all_records()
        if records:
            return pd.DataFrame(records)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return pd.DataFrame()


def save_label_to_sheet(sheet, message_id, message, sentiment, confidence, labeler_name, timestamp):
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
st.markdown("Label Twitch chat messages by sentiment.")

# sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    labeler_name = st.text_input("Your Name:", placeholder="e.g., Student 1")

    st.divider()

    # google Sheets Connection
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

    # load dataset
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

    with st.expander("Excitement"):
        st.write("""
        **Positive emotion about gameplay**
        - Examples: POGGERS, LETS GO, CLUTCH, HOLY
        - Signs: Caps lock, !, positive gaming emotes
        """)

    with st.expander("Frustration"):
        st.write("""
        **Negative emotion, disappointment**
        - Examples: wtf, trash, ff15, throw
        - Signs: Curse words, negative words, ?
        """)

    with st.expander("Humor"):
        st.write("""
        **Jokes, memes, sarcasm, playfulness**
        - Examples: KEKW, copypasta, laugh emotes
        - Signs: Laugh emotes, joke structure, sarcasm
        """)

    with st.expander("Confusion"):
        st.write("""
        **Questions, not understanding**
        - Examples: what happened?, ???, how?
        - Signs: Question marks, confusion emotes
        """)

    with st.expander("Boredom"):
        st.write("""
        **Lack of interest, slow pace**
        - Examples: ResidentSleeper, zzzz, boring
        - Signs: Sleep emotes, pace complaints
        """)

    with st.expander("Neutral"):
        st.write("""
        **Everything else**
        - Examples: hi, gg, nice, general chat
        - Signs: Informational, greetings, generic
        """)

# main
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


    # load 7TV emotes from API
    @st.cache_resource
    def load_7tv_emotes():
        import requests

        # hard coded popular emotes as fallback
        popular_emotes = {
            'POGGERS': '60aeb9d4b55b1d4d87f0a8a5',
            'PogChamp': '604e18e3d1b8e6471b86d6d8',
            'Pog': '60aeb9d4b55b1d4d87f0a8a5',
            'KEKW': '60aeb9d4b55b1d4d87f0a8a3',
            'LUL': '604bce0e5c4505c4e4f0a0b7',
            'Sadge': '60aeb9d4b55b1d4d87f0a8a2',
            'FeelsBadMan': '60aeb9d4b55b1d4d87f0a8a1',
            'ResidentSleeper': '60aeb9d4b55b1d4d87f0a8a4',
            'Pepega': '60aeb9d4b55b1d4d87f0a8a6',
            'OMEGALUL': '60aeb9d4b55b1d4d87f0a8a7',
            'Kappa': '6074b3c6e7c4b5a1d2e3f4g5',
            'MonkaS': '60aeb9d4b55b1d4d87f0a8a8',
            'Clap': '60aeb9d4b55b1d4d87f0a8a9',
            'CoolStoryBob': '61f6a74f0c6d9e8a5b4c3d2e',
            'PepeHands': '60aeb9d4b55b1d4d87f0a8aa',
            'Wicked': '60aeb9d4b55b1d4d87f0a8ab',
            'Weirdge': '60aeb9d4b55b1d4d87f0a8ac',
            'Thonk': '60aeb9d4b55b1d4d87f0a8ad',
            'AYAYA': '60aeb9d4b55b1d4d87f0a8ae',
            'TrollDespair': '60aeb9d4b55b1d4d87f0a8af',
            'Copium': '60aeb9d4b55b1d4d87f0a8b0',
            'BASED': '60aeb9d4b55b1d4d87f0a8b1',
            ' Monkas': '60aeb9d4b55b1d4d87f0a8b2',
            'NODDERS': '60aeb9d4b55b1d4d87f0a8b3',
            'YEAHBUT': '60aeb9d4b55b1d4d87f0a8b4',
        }

        try:
            # try to fetch from 7TV API
            response = requests.get('https://api.7tv.app/v3/emote-sets/global', timeout=5)
            if response.status_code == 200:
                data = response.json()
                emotes = {}
                if 'emotes' in data:
                    for emote in data['emotes'][:100]:  # top 100 emotes
                        emotes[emote['name']] = emote['id']
                # merge with popular emotes
                emotes.update(popular_emotes)
                return emotes
        except:
            pass

        # return popular emotes if API fails
        return popular_emotes


    emote_map_7tv = load_7tv_emotes()


    def get_7tv_emote_url(emote_id):
        """Get 7TV emote image URL"""
        return f"https://cdn.7tv.app/emotes/{emote_id}/4x.webp"


    def render_message_with_emotes(text):
        """Render message with 7TV emote images"""
        html = f'<div style="font-size: 18px; line-height: 1.8;">'
        words = text.split()

        for word in words:
            if word in emote_map_7tv:
                emote_id = emote_map_7tv[word]
                emote_url = get_7tv_emote_url(emote_id)
                html += f'<img src="{emote_url}" alt="{word}" style="height: 28px; margin: 0 2px; vertical-align: middle;">'
            else:
                html += f'<span style="margin-right: 4px;">{word}</span>'

        html += '</div>'
        return html


    # display current message
    if st.session_state.current_message:
        st.divider()

        st.markdown("### 💬 Current Message")
        message_container = st.container(border=True)
        with message_container:
            st.write(f"**ID:** {st.session_state.message_index}")
            # Display message with 7TV emotes
            emote_html = render_message_with_emotes(st.session_state.current_message)
            st.markdown(emote_html, unsafe_allow_html=True)
            st.caption(f"Original text: {st.session_state.current_message}")

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

    # progress tracker
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

    # show all labeled data from sheet
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

                # sentiment distribution
                st.subheader("Sentiment Distribution (All)")
                sentiment_counts = df_all['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

                # labeler
                st.subheader("Labels by Team Member")
                labeler_counts = df_all['labeled_by'].value_counts()
                st.bar_chart(labeler_counts)

                # recent labels
                st.subheader("Recent Labels (Latest 15)")
                st.dataframe(df_all.tail(15).iloc[::-1], use_container_width=True)
            else:
                st.info("No labels yet. Start labeling!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Twitch Sentiment Labeling | CS 175 Project<br>
Labels synced to Google Sheets.
</div>
""", unsafe_allow_html=True)