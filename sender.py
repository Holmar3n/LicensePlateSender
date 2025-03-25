from flask import Flask, jsonify, Response
import requests
import cv2
import numpy as np
import time
import threading
from picamera2 import Picamera2
import io
from PIL import Image, ImageEnhance
from rpi_rf import RFDevice
import RPi.GPIO as GPIO
from ultralytics import YOLO

app = Flask(__name__)

GPIO.setmode(GPIO.BCM)

RF_TX_PIN = 17 # GPIO FÖR SÄNDAREN
code_open = 954500 # RELÄ KODEN FÖR ATT ÖPPNA
code_close = 154388 # RELÄ KODEN FÖR ATT STÄNGA
pulselength = None # PULSLÄNGDEN FRÅN MOTTAGAREN
TIME_OPEN = 10.5 # TIDEN DET TAR ATT ÖPPNA GRINDEN
TIME_CLOSE = 30
GATE_MOVEMENT_TIME = 0

GATE_MOVING = False
GATE_STATUS = "Closed"
STOP_SIGNAL = False
CAPTURE_TIME = 40

RTSP_URL = "rtsp://anders.hellring@gmail.com:Sae6uakatt@192.168.68.125:554/stream2"
RTSP_URL_HIGH = "rtsp://anders.hellring@gmail.com:Sae6uakatt@192.168.68.125:554/stream1"

SERVER_URL = "http://85.229.92.237:8080/AnalyzePicture"

model = YOLO("yolov8n.pt")

#picam2 = Picamera2()

#camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)})
#picam2.configure(camera_config)
#picam2.set_controls({"AwbMode": 1, "AnalogueGain": 1.0, "ExposureTime": 50000})

capture_running = False

# Flagga för att kontrollera om bildtagningen ska köras
IP_CAMERA_RUL = "rtsp://anders.hellring@gmail.com:Sae6uakatt@192.168.68.125:554/stream1"
camera_active = False

def generate_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Kunde inte ansluta till IP kamera")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        
        yield (b"--frame\r\n"
               b"Content-Type_ image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        
    cap.release()


capture_running = False

def get_remaining_time():
    return max(0, TIME_OPEN - GATE_MOVEMENT_TIME)

def capture_image():
    """Hämtar en bild från IP-kameran via RTSP, beskär, förbättrar och komprimerar den för OCR."""
    cap = cv2.VideoCapture(RTSP_URL_HIGH)

    if not cap.isOpened():
        print("❌ Kunde inte ansluta till IP-kameran.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Misslyckades att hämta bild från IP-kameran.")
        return None

    # ✂️ Beskär nedre mitten
    height, width, _ = frame.shape
    top = int(height * 0.4)
    bottom = height
    left = int(width * 0.4)
    right = int(width * 0.9)
    cropped_frame = frame[top:bottom, left:right]

    # 🛠 Förbättra kontrast & skärpa
    image_pil = Image.fromarray(cropped_frame[..., ::-1])  # BGR → RGB
    image_pil = ImageEnhance.Contrast(image_pil).enhance(1.2)
    image_pil = ImageEnhance.Sharpness(image_pil).enhance(1.3)

    # ⬇️ Skala ner till 800x600
    image_pil = image_pil.resize((800, 600), Image.Resampling.LANCZOS)

    # 💾 Komprimera till JPEG med kvalitet 75
    buffer = io.BytesIO()
    image_pil.save(buffer, format="JPEG", quality=60)
    return buffer.getvalue()

def detect_license_plate(image):
    """Detekterar fordon i bilden och filtrerar baserat på storlek (närhet)."""
    results = model(image)
    vehicle_classes = [2, 5]  # 2 = bil, 5 = buss
    MIN_AREA = 30000  # Justera beroende på hur nära bilen ska vara

    vehicle_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  # Klass-ID

            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                print(f"🔍 Fordon hittat med area: {area}")

                if area < MIN_AREA:
                    print("🚫 Fordonet är för långt bort, hoppar över.")
                    continue

                label = "Bil" if cls == 2 else "Buss"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                vehicle_detected = True

    return vehicle_detected
def capture_and_send():
    """Fortsätter ta bilder från IP-kameran tills en registreringsskylt identifieras eller en timeout inträffar."""
    global capture_running , CAPTURE_TIME

    print("📸 Startar bildtagning från IP-kamera...")
    
    start_time = time.time()
    while time.time() - start_time < CAPTURE_TIME and capture_running:
        try:
            image_data = capture_image()
            if image_data is None:
                print("⚠️ Kunde inte hämta bild, försöker igen...")
                continue

            img_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # 🛠 Spara originalbilden för att debugga
            cv2.imwrite("debug_original.jpg", image)
        

            if detect_license_plate(image):
                print("✅ Skylt identifierad! Skickar bild till servern...")
                files = {"file": ("camera.jpg", image_data, "image/jpeg")}
                response = requests.post(SERVER_URL, files=files)

                if response.status_code == 200:
                    result = response.json()
                    print(f"📩 Svar från server: {result}")

                    if result.get("status") == "valid":
                        print("🚗 Skylten är godkänd! Öppnar grinden och stoppar bildtagning.")
                        requests.post("http://82.196.100.136:5001/OpenGate")
                        capture_running = False
                    elif result.get("status") == "invalid" and result.get("message") == "Kunde inte identifiera ett registreringsnummer.":
                        print("❌ Ingen registreringsskylt identifierad, fortsätter skanna...")
                        capture_running = True
                        
                    else:
                        print("❌ Skylten är ej godkänd! Stoppar bildtagning.")
                        capture_running = False
                else:
                    print("⚠️ Servern kunde inte identifiera en skylt. Fortsätter bildtagning...")

            else:
                print("❌ Ingen registreringsskylt identifierad, fortsätter skanna...")

        except Exception as e:
            print(f"⚠️ Fel vid bildhantering: {e}")

        elapsed_time = time.time() - start_time
        sleep_time = max(0, 1.5 - elapsed_time)
        time.sleep(sleep_time)
    capture_running = False
    print("📴 Bildtagningen har stoppats.")
    

@app.route('/start_capture', methods=['POST'])
def start_capture():
    """Startar bildtagningen om den inte redan körs."""
    global capture_running
    
    if capture_running:
        return jsonify({"message": "Bildtagning körs redan!"}), 400

    capture_running = True  
    threading.Thread(target=capture_and_send, daemon=True).start()
        
    return jsonify({"message": "Bildtagning startad!"})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    """Stoppar bildtagningen manuellt."""
    global capture_running  
    capture_running = False  
    return jsonify({"message": "Bildtagning stoppad!"})

@app.route("/OpenGate", methods=["POST"])
def open_gate():
    global GATE_MOVING, GATE_STATUS, STOP_SIGNAL, GATE_MOVEMENT_TIME

    if GATE_MOVING:
        STOP_SIGNAL = True  # ⛔️ Om du trycker igen, stoppa loopen
        time.sleep(0.5)
        elapsed_time = time.time() - start_time if "start_time" in globals() else 0
        GATE_MOVEMENT_TIME = max(0, GATE_MOVEMENT_TIME - elapsed_time)
        STOP_SIGNAL = True
        return jsonify("Grinden rör sig redan"), 200
    remaining_time = get_remaining_time()
    if remaining_time <= 0:
        return jsonify("Grinden är redan helt öppen!"), 400
    
    rfdevice = RFDevice(RF_TX_PIN)
    rfdevice.enable_tx()

    try:
        print("Skickar...")
        start_time = time.time()
        GATE_MOVING = True  

        while time.time() - start_time < remaining_time:
            if STOP_SIGNAL:  # ✅ Om du trycker igen, avbryt loopen
                print("Grinden stoppas!")
                break

            rfdevice.tx_code(code_open)
            time.sleep(0.001)
                   
        elapsed_time = time.time() - start_time
        GATE_MOVEMENT_TIME = min(TIME_OPEN, GATE_MOVEMENT_TIME + elapsed_time)


    finally:
        STOP_SIGNAL = False 
        GATE_MOVING = False
        rfdevice.cleanup()

    return jsonify(f"Grinden har öppnats i {elapsed_time:.2f} sekunder!"), 200

@app.route("/CloseGate", methods=["POST"])
def close_gate():
    global GATE_MOVING, STOP_SIGNAL, GATE_MOVEMENT_TIME

    rfdevice = RFDevice(RF_TX_PIN)
    rfdevice.enable_tx()

    if GATE_MOVING:
        STOP_SIGNAL = True
        return jsonify("Grinden rör sig redan"), 200

    try:
        print("Stänger grinden...")
        start_time = time.time()
        GATE_MOVING = True

        while time.time() - start_time < TIME_CLOSE:  # ?? Stäng alltid i 15 sekunder
            if STOP_SIGNAL:
                elapsed_time = time.time() - start_time
                GATE_MOVEMENT_TIME = max(0, GATE_MOVEMENT_TIME - (elapsed_time * (TIME_OPEN/TIME_CLOSE)))  # ?? Justera till 12-sekunders skala
                print(f"Grinden stoppades efter {elapsed_time:.2f} sekunder. Justerad öppningstid: {GATE_MOVEMENT_TIME:.2f} sekunder")
                break
            rfdevice.tx_code(code_close)
            time.sleep(0.001)

        # ?? Om hela stängningen körs utan avbrott, nollställ tiden
        if time.time() - start_time >= TIME_CLOSE:
            elapsed_time = 0
            GATE_MOVEMENT_TIME = 0
            print("Grinden har stängts helt, återställ elapsed_time och GATE_MOVEMENT_TIME.")

    finally:
        STOP_SIGNAL = False
        GATE_MOVING = False
        rfdevice.cleanup()

    return jsonify(f"Grinden har stängts! Justerad öppningstid: {GATE_MOVEMENT_TIME:.2f} sekunder"), 200

#def generate_frames():
#    print("Livestream startad!")
#    picam2.start()
#    try:
    
#        while True:
#            frame = picam2.capture_array()
#            ret, buffer = cv2.imencode(".jpg", frame)
#            frame = buffer.tobytes()
#            yield (b"--frame\r\n"
#                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
#            time.sleep(1/15)
#    except GeneratorExit:
#        picam2.stop()
#        print("Livestream Avbruten!")
@app.route("/LiveStream")
def live_stream():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "_main_":
    app.run(host="0.0.0.0", port=5001, threaded=True)
