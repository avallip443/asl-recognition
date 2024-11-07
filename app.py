from flask import Flask, Response
import cv2
import os

# Initialize the Flask app
app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)

# Directory to save captured images
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def index():
    variable = 12345
    html_content = '''
        <html>
            <head>
                <title>ASL to Text</title>
            </head>
            <body>
                <h1>ASL to Text</h1>
                <img src="/video_feed" width="1080" height="620" />
                <br><br>
                <a href="/capture"><button>Capture Image</button></a>
                <p>Conversion: {} </p>
            </body>
        </html>
    '''.format(variable)
    return html_content

# Video stream route
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Yield the frame as a byte stream for the live feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture_image():
    ret, frame = cap.read()
    if ret:
        # Save the captured image to the data directory
        file_path = os.path.join(DATA_DIR, 'captured_image.jpg')
        cv2.imwrite(file_path, frame)
        return f"Image saved to {file_path}"
    else:
        return "Failed to capture image."

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
