import cv2
import mediapipe as mp
import numpy as np
import pickle
import socket
from test_classifier import generate
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Render the home page with dynamic video feed."""
    return render_template('index.html')

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port assigned by the OS
        return s.getsockname()[1]
    
@app.route('/video_feed')
def video_feed():
    """Generate video feed for the web page."""
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        port = get_free_port()
        app.run(debug=True, host='0.0.0.0', port=port)
    finally:
        # Ensure camera release at server shutdown
        try:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed at server shutdown.")
        except Exception as e:
            print(f"Error during shutdown cleanup: {e}")
