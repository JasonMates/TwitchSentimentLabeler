import streamlit as st
import pandas as pd
import random
from datetime import datetime
from datasets import load_dataset
import os

st.set_page_config(page_title="Twitch Sentiment Labeler", layout="centered")

# init session state
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


# load dataset
@st.cache_resource
def load_twitch_data():
    """load and cache the Twitch dataset from huggingface"""
    try:
        dataset = load_dataset("lparkourer10/twitch_chat")
        # extract just the message text from the dataset
        messages = [msg.get('message', msg.get('text', str(msg))) for msg in dataset['train']]
        return messages
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []


# title and description
st.title("Twitch Chat Sentiment Labeler")
st.markdown("Label Twitch chat messages by sentiment.")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    labeler_name = st.text_input("Your Name:", placeholder="e.g., Student 1")

    st.divider()

    # load dataset info
    if st.button("📥 Load Dataset", use_container_width=True):
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

# main content
if not st.session_state.dataset_loaded:
    st.warning("⚠️ Click 'Load Dataset' in the sidebar to begin!")
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

    # display current message
    if st.session_state.current_message:
        st.divider()

        # message display
        st.markdown("### 💬 Current Message")
        message_container = st.container(border=True)
        with message_container:
            st.write(f"**ID:** {st.session_state.message_index}")
            st.markdown(f'```\n{st.session_state.current_message}\n```')

        st.divider()

        # labeling interface
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

        # submit button
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if st.button("✅ Submit Label", use_container_width=True, type="primary"):
                if sentiment != "Select..." and confidence != "Select...":
                    # Extract confidence score (first character)
                    confidence_score = int(confidence[0])

                    # prepare entry
                    entry = pd.DataFrame({
                        'message_id': [st.session_state.message_index],
                        'message': [st.session_state.current_message],
                        'sentiment': [sentiment],
                        'confidence': [confidence_score],
                        'labeled_by': [labeler_name],
                        'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                    })

                    # save to csv
                    csv_filename = 'twitch_labels.csv'
                    try:
                        df = pd.read_csv(csv_filename)
                        df = pd.concat([df, entry], ignore_index=True)
                    except FileNotFoundError:
                        df = entry

                    df.to_csv(csv_filename, index=False)

                    # success message
                    st.success(f"✅ Labeled as **{sentiment}** (Confidence: {confidence_score}/5)")
                    st.session_state.labeled_count += 1
                    st.balloons()

                    # progress
                    st.info(f"📊 You've labeled **{st.session_state.labeled_count} messages** so far!")

                    # clear for next message
                    st.session_state.current_message = None
                    st.session_state.message_index = None

                else:
                    st.error("⚠️ Please select both sentiment and confidence!")
    else:
        if st.session_state.dataset_loaded:
            st.info("Click 'Load Random Message' to start labeling!")

    # progress tracker
    st.divider()
    st.markdown("### Labeling Progress")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messages Labeled", st.session_state.labeled_count, "this session")
    with col2:
        st.metric("Target", 500, "for your team member")
    with col3:
        progress_pct = min((st.session_state.labeled_count / 500) * 100, 100)
        st.metric("Progress", f"{progress_pct:.1f}%")

    # progress bar
    progress_bar = st.progress(min(st.session_state.labeled_count / 500, 1.0))

    # labeled data
    st.divider()
    st.markdown("### 💾 Your Labeled Messages")
    csv_filename = 'twitch_labels.csv'
    if os.path.exists(csv_filename):
        df_labeled = pd.read_csv(csv_filename)
        df_labeled_user = df_labeled[df_labeled['labeled_by'] == labeler_name] if labeler_name else df_labeled

        st.write(f"**Total labeled by {labeler_name if labeler_name else 'all'}:** {len(df_labeled_user)}")

        # sentiment distribution
        if len(df_labeled_user) > 0:
            sentiment_counts = df_labeled_user['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            # show recent labels
            st.subheader("Recent Labels")
            st.dataframe(df_labeled_user.tail(10).iloc[::-1], use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Twitch Sentiment Labeling Tool | CS 175 Project<br>
<br>
Labels saved to: twitch_labels.csv
</div>
""", unsafe_allow_html=True)