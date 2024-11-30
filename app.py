import cv2
import socket
from prediction import generate
from flask import Flask, render_template, Response,jsonify
from prediction import cap,get_character
import webbrowser
import logging
import time

dynamic_data = ""
updated_stream=""
new_char = ""
last_detected_char = ""

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.globals['__name__'] = '__main__'  # Disable template caching
app.jinja_env.cache = {}

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

# A route that will return the data for the text box
@app.route('/get_dynamic_data')
def get_dynamic_data():
    global updated_stream, dynamic_data,new_char,last_detected_char
    current_time = time.time()
    new_char = get_character()
    if last_detected_char != new_char: 
        updated_stream += str(new_char)
        dynamic_data = updated_stream
        last_detected_char = new_char
        return jsonify({"dynamic_text": dynamic_data})
    return ""

# A route to reset the data on the server side (clear dynamic data)
@app.route('/clear_dynamic_data')
def clear_dynamic_data():
    global updated_stream, dynamic_data
    updated_stream = ""  # Reset the 'updated_stream' variable
    dynamic_data = ""  # Reset the dynamic data
    logging.info("Cleared dynamic data on the server.")
    return jsonify({"message": "Dynamic data cleared"})

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

    finally:
        # Ensure camera release at server shutdown
        try:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed at server shutdown.")
        except Exception as e:
            print(f"Error during shutdown cleanup: {e}")
