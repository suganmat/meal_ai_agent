"""
Main entry point for the Meal Prediction AI Agent application.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meal_ai_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check if required environment variables are set."""
    required_vars = [
        'PERPLEXITY_API_KEY',
        'NEO4J_URI',
        'NEO4J_USERNAME',
        'NEO4J_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file or environment")
        return False
    
    return True


def run_streamlit_app():
    """Run the Streamlit chat application."""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Run Streamlit app
        sys.argv = [
            'streamlit',
            'run',
            'interface/chat_app.py',
            '--server.port=8501',
            '--server.address=localhost'
        ]
        
        stcli.main()
        
    except Exception as e:
        logger.error(f"Error running Streamlit app: {e}")
        raise


def run_cli_mode():
    """Run the application in CLI mode for testing."""
    try:
        from src.services.langgraph_orchestrator import MealAgentOrchestrator
        
        logger.info("Starting Meal AI Agent in CLI mode...")
        logger.info("Type 'quit' to exit")
        
        # Initialize orchestrator
        orchestrator = MealAgentOrchestrator()
        session_id = orchestrator.session_manager.create_session()
        
        print("\n🍽️ AI Meal Recommendation Assistant")
        print("=" * 50)
        print("Hello! I'm here to help you find perfect meals.")
        print("Type 'quit' to exit at any time.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("AI: Goodbye! Enjoy your meals! 🍽️")
                    break
                
                if not user_input:
                    continue
                
                # Process message
                response = orchestrator.process_message(user_input, session_id)
                print(f"AI: {response}\n")
                
            except (EOFError, KeyboardInterrupt):
                print("\nAI: Goodbye! Enjoy your meals! 🍽️")
                break
            except Exception as e:
                logger.error(f"Error in CLI mode: {e}")
                print(f"AI: I'm sorry, I encountered an error: {e}")
                break
        
    except Exception as e:
        logger.error(f"Error initializing CLI mode: {e}")
        raise


def main():
    """Main function to run the application."""
    logger.info("Starting Meal Prediction AI Agent...")
    
    # Check environment variables
    if not check_environment():
        logger.error("Environment check failed. Please fix the issues above.")
        return 1
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'cli':
            run_cli_mode()
        elif mode == 'streamlit':
            run_streamlit_app()
        else:
            print("Usage: python main.py [cli|streamlit]")
            print("  cli       - Run in command line interface mode")
            print("  streamlit - Run Streamlit web interface (default)")
            return 1
    else:
        # Default to Streamlit
        run_streamlit_app()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
