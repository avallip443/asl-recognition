import cv2
import mediapipe as mp
import numpy as np
import pickle
import socket
from prediction import generate
from flask import Flask, render_template, Response, send_file
import os
from prediction import cap
import webbrowser

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.globals['__name__'] = '__main__'  # Disable template caching
app.jinja_env.cache = {}

@app.route('/')
def index():
    print("Rendering index.html")
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
               # URL to open
        url = "http://127.0.0.1:"+ str(port)
        webbrowser.open(url)
        # Open the URL in the default web browser
        app.run(debug=True, host='0.0.0.0', port=port, use_reloader = False)
        app.run(debug=True, host='0.0.0.0', port=port)
    finally:
        # Ensure camera release at server shutdown
        try:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed at server shutdown.")
        except Exception as e:
            print(f"Error during shutdown cleanup: {e}")
