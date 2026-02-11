"""
AI Customer Service Chatbot — application entry point.

Usage:
    python run.py
"""
from chatbot import create_app
from chatbot.config import PORT

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
