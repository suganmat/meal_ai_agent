"""
Streamlit chat interface for the meal prediction AI agent.
"""
import streamlit as st
import uuid
import os
from typing import Dict, Any, List
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the src directory to the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.services.langgraph_orchestrator import MealAgentOrchestrator

# Configure Streamlit page
st.set_page_config(
    page_title="🍽️ AI Meal Recommendation Assistant",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
    }
    
    /* Chat container with proper height */
    .chat-container {
        height: 50vh;
        max-height: 50vh;
        background: #fafafa;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        margin-top: 2rem;
    }
    
    /* Chat messages area - grows upward */
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        background: white;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        min-height: 0;
    }
    
    /* Chat input area - fixed at bottom */
    .chat-input-area {
        background: white;
        border-top: 1px solid #e2e8f0;
        padding: 1rem;
        flex-shrink: 0;
    }
    
    /* Ensure messages start from bottom and grow up */
    .chat-messages .stChatMessage {
        margin-bottom: 1rem;
    }
    
    /* Auto-scroll to bottom */
    .chat-messages {
        scroll-behavior: smooth;
    }
    
    /* Chat messages */
    .stChatMessage {
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.3s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 1rem 1.2rem;
        margin-left: 20%;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant"] {
        background: white;
        color: #333;
        border-radius: 18px 18px 18px 4px;
        padding: 1rem 1.2rem;
        margin-right: 20%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e1e5e9;
        padding: 0.8rem 1.2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .session-info {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .session-info h3 {
        color: #4a5568;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #667eea;
    }
    
    /* Progress indicators */
    .progress-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: #f7fafc;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .progress-item:hover {
        background: #edf2f7;
        transform: translateX(4px);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-active {
        background: #c6f6d5;
        color: #22543d;
    }
    
    .status-complete {
        background: #bee3f8;
        color: #2a4365;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        background: #f7fafc;
        border-radius: 8px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .stChatMessage[data-testid="user"] {
            margin-left: 10%;
        }
        
        .stChatMessage[data-testid="assistant"] {
            margin-right: 10%;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "orchestrator" not in st.session_state:
        try:
            st.session_state.orchestrator = MealAgentOrchestrator()
        except Exception as e:
            st.error(f"Failed to initialize AI agent: {e}")
            st.session_state.orchestrator = None
    
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {
            "total_messages": 0,
            "session_start": time.time()
        }


def display_chat_history():
    """Display chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_session_info():
    """Display session information in sidebar."""
    with st.sidebar:
        st.markdown("### 📊 Session Information")
        
        session_info = st.session_state.session_stats
        duration = int(time.time() - session_info['session_start'])
        
        # Create a more visually appealing session info card
        st.markdown(f"""
        <div class="session-info">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="width: 8px; height: 8px; background: #48bb78; border-radius: 50%; margin-right: 0.5rem;"></div>
                <strong>Session Active</strong>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <span style="color: #718096; font-size: 0.9rem;">ID:</span> 
                <code style="background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.8rem;">{st.session_state.session_id[:8]}...</code>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <span style="color: #718096; font-size: 0.9rem;">Messages:</span> 
                <strong style="color: #2d3748;">{session_info['total_messages']}</strong>
            </div>
            <div>
                <span style="color: #718096; font-size: 0.9rem;">Duration:</span> 
                <strong style="color: #2d3748;">{duration}s</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display current state if orchestrator is available
        if st.session_state.orchestrator:
            try:
                state = st.session_state.orchestrator.get_session_state(st.session_state.session_id)
                current_state = state.get('current_state', 'initial')
                
                # Status badge
                status_color = "status-active" if current_state != "initial" else "status-complete"
                st.markdown(f"""
                <div style="margin-bottom: 1.5rem;">
                    <h4 style="margin-bottom: 0.5rem; color: #4a5568;">🔄 Current Status</h4>
                    <span class="status-badge {status_color}">{current_state.replace('_', ' ').title()}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if state.get('user_id'):
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <span style="color: #718096; font-size: 0.9rem;">User ID:</span> 
                        <code style="background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.8rem;">{state['user_id'][:8]}...</code>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display profile collection progress with better styling
                profile = state.get('profile_collection', {})
                if profile:
                    st.markdown("### 👤 Profile Progress")
                    
                    progress_items = []
                    if profile.get('name'):
                        progress_items.append(("Name", True))
                    if profile.get('age'):
                        progress_items.append(("Age", True))
                    if profile.get('primary_cuisine'):
                        progress_items.append(("Primary Cuisine", True))
                    if profile.get('secondary_cuisine'):
                        progress_items.append(("Secondary Cuisine", True))
                    if profile.get('medical_conditions'):
                        progress_items.append(("Medical Conditions", True))
                    
                    # Create progress items with better styling
                    for item_name, is_complete in progress_items:
                        icon = "✅" if is_complete else "⏳"
                        color = "#48bb78" if is_complete else "#a0aec0"
                        st.markdown(f"""
                        <div class="progress-item">
                            <span style="color: {color}; margin-right: 0.5rem;">{icon}</span>
                            <span style="color: #4a5568;">{item_name}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if profile.get('is_complete'):
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #48bb78, #38a169); color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;">
                            <strong>🎉 Profile Complete!</strong>
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error getting session state: {e}")


def display_sidebar_controls():
    """Display sidebar controls."""
    with st.sidebar:
        st.markdown("### 🎛️ Controls")
        
        # Create button container with better styling
        col1, col2 = st.columns(2)
        
        with col1:
            # Clear session button
            if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.clear_session(st.session_state.session_id)
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.session_stats = {
                    "total_messages": 0,
                    "session_start": time.time()
                }
                st.rerun()
        
        with col2:
            # New session button
            if st.button("🆕 New", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.session_stats = {
                    "total_messages": 0,
                    "session_start": time.time()
                }
                st.rerun()
        
        st.markdown("---")
        
        # Display help information with better styling
        st.markdown("### 💡 How to Use")
        
        help_steps = [
            ("👋", "Introduce yourself", "Tell me your name"),
            ("🍽️", "Share preferences", "Age, cuisine, health conditions"),
            ("🎯", "Get suggestions", "I'll recommend personalized meals"),
            ("💬", "Provide feedback", "Let me know what you think")
        ]
        
        for icon, title, description in help_steps:
            st.markdown(f"""
            <div style="display: flex; align-items: flex-start; margin-bottom: 1rem; padding: 0.75rem; background: #f8fafc; border-radius: 8px; border-left: 3px solid #667eea;">
                <span style="font-size: 1.2rem; margin-right: 0.75rem; margin-top: 0.1rem;">{icon}</span>
                <div>
                    <div style="font-weight: 600; color: #2d3748; margin-bottom: 0.25rem;">{title}</div>
                    <div style="font-size: 0.9rem; color: #718096;">{description}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem; text-align: center;">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">💬 General Chat</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Ask about cooking, nutrition, or anything else!</div>
        </div>
        """, unsafe_allow_html=True)


def process_user_input(user_input: str) -> str:
    """Process user input and return AI response."""
    if not st.session_state.orchestrator:
        return "I'm sorry, the AI agent is not available. Please refresh the page."
    
    try:
        # Process message through orchestrator
        response = st.session_state.orchestrator.process_message(
            user_input, 
            st.session_state.session_id
        )
        
        # Update session stats
        st.session_state.session_stats["total_messages"] += 1
        
        return response
        
    except Exception as e:
        st.error(f"Error processing message: {e}")
        return "I'm sorry, I encountered an error processing your message. Please try again."


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header with improved styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h1 class="main-header">🍽️ AI Meal Recommendation Assistant</h1>
        <p style="color: #718096; font-size: 1.1rem; margin-top: 0.5rem;">
            Your personal AI chef for personalized meal recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat messages area (grows upward)
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.messages:
        display_chat_history()
    else:
        # Welcome message for new users
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #718096;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">👋</div>
            <h3 style="color: #4a5568; margin-bottom: 1rem;">Welcome to your AI Meal Assistant!</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                I can help you with personalized meal recommendations or answer any questions you have. 
                Just start typing below!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close messages area
    
    # Chat input area (fixed at bottom)
    st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
    
    # Chat input with improved styling
    if prompt := st.chat_input("💬 Ask me about meals or anything else...", key="main_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display AI response with better loading state
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = process_user_input(prompt)
            
            # Add typing animation effect
            st.markdown(response)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the display
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close input area
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat container
    
    # Add JavaScript for auto-scroll
    st.markdown("""
    <script>
    function scrollToBottom() {
        const chatMessages = document.querySelector('.chat-messages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // Scroll to bottom when page loads
    window.addEventListener('load', scrollToBottom);
    
    // Scroll to bottom when new messages are added
    const observer = new MutationObserver(scrollToBottom);
    const chatContainer = document.querySelector('.chat-messages');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        display_session_info()
        display_sidebar_controls()
    
    # Display footer with improved styling
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
            <span>🤖 Powered by AI</span>
            <span>•</span>
            <span>⚡ Built with Streamlit</span>
            <span>•</span>
            <span>🍽️ Meal Recommendation System</span>
        </div>
        <div style="font-size: 0.8rem; opacity: 0.7;">
            Made with ❤️ for better meal planning
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
