import streamlit as st
from rag_pipeline import stream_llm_rag_response
from langchain_openai import OpenAI
import dotenv
import os

# Page configuration
st.set_page_config(
    page_title="Bulking Meals Chatbot",
    page_icon="💪",
    layout="centered"
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'should_process' not in st.session_state:
    st.session_state.should_process = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .stButton>button {
        border-radius: 20px;
    }
    
    /* Scrollable chat container */
    .stChatMessageContent {
        overflow-wrap: break-word;
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# App header
st.title("💪 Bulking Meals Chatbot")
st.write("Ask me about high-protein meals and fitness nutrition!")

# Function to set flag for processing
def set_process_flag():
    st.session_state.should_process = True

# Process messages if flag is set
if st.session_state.should_process and st.session_state.user_input:
    user_message = st.session_state.user_input
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Clear input and reset flag
    st.session_state.user_input = ""
    st.session_state.should_process = False
    
    # Use RAG pipeline
    llm = OpenAI(api_key=api_key, temperature=0.7)
    rag_messages = st.session_state.messages.copy()
    response_generator = stream_llm_rag_response(llm, rag_messages)
    
    # Process the response
    for _ in response_generator:
        pass

# Display chat messages using Streamlit's built-in chat components
chat_container = st.container(height=500)  # Fixed height container

# Use Streamlit's built-in chat message components
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="💪"):
                st.markdown(message["content"])

# Chat input area - at the bottom of the screen
with st.container():
    # Create a horizontal layout for input and button
    cols = st.columns([4, 1])
    
    # Input field
    with cols[0]:
        st.text_input(
            "Ask about meals and nutrition:",
            key="user_input",
            placeholder="e.g., What's a high protein breakfast?",
            label_visibility="collapsed"
        )
    
    # Send button
    with cols[1]:
        if st.button("Send", use_container_width=True, on_click=set_process_flag) or (
            st.session_state.user_input and st.session_state.user_input.endswith("\n")
        ):
            set_process_flag()
            st.rerun()

# Add some helpful example prompts
with st.expander("Example questions you can ask"):
    st.markdown("""
    - What are good high-protein breakfast options?
    - I need meal ideas for bulking with 3000 calories
    - What protein-rich foods can I eat as a vegetarian?
    - Give me a weekly meal plan for muscle building
    - What should I eat before and after a workout?
    """)

# Auto-scroll to bottom using JavaScript
st.markdown("""
<script>
    // Function to scroll to the bottom
    function scrollChatToBottom() {
        const chatContainer = document.querySelector('[data-testid="stVerticalBlock"] > div');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
            console.log("Scrolled to bottom");
        }
    }
    
    // Attempt to scroll after the page is loaded
    window.addEventListener('load', function() {
        setTimeout(scrollChatToBottom, 500);  // Delay to ensure content is rendered
    });
    
    // Try to set up a mutation observer to detect new messages
    const observer = new MutationObserver(function(mutations) {
        scrollChatToBottom();
    });
    
    // Start observing when document is fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        const chatContainer = document.querySelector('[data-testid="stVerticalBlock"] > div');
        if (chatContainer) {
            observer.observe(chatContainer, { 
                childList: true,
                subtree: true 
            });
            console.log("Observer set up");
        }
    });
</script>
""", unsafe_allow_html=True)




