from flask import Flask, jsonify
import requests
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)

IP_WEBCAM_URL = "http://192.168.1.174:8080/shot.jpg"
SERVER_URL = "http://127.0.0.1:8080/AnalyzePicture"
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Flagga för att kontrollera om bildtagningen ska köras
capture_running = False  

def detect_license_plate(image):
    """Identifierar en registreringsskylt i bilden."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 40))

    return len(plates) > 0

def capture_and_send():
    """Fortsätter ta bilder tills servern identifierar en skylt och ger ett svar."""
    global capture_running  

    print("📡 Sendern startar bildtagning...")

    while capture_running:
        start_time = time.time()

        try:
            response = requests.get(IP_WEBCAM_URL, stream=True, timeout=5)
            if response.status_code == 200:
                img_data = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                if detect_license_plate(image):
                    print("🚗 Skylt identifierad! Skickar bild till servern...")
                    files = {"file": ("webcam.jpg", response.content, "image/jpeg")}
                    response = requests.post(SERVER_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()
                        print(f"📡 Svar från server: {result}")

                        if "status" in result:
                            if result["status"] == "valid":
                                print("✅ Skylten är godkänd! Öppnar grinden och stoppar bildtagning.")
                                requests.post("http://127.0.0.1:8080/OpenGate")
                                capture_running = False  # 🛑 Stoppar bildtagningen

                            elif result["status"] == "invalid" and result["message"] == "Registreringsnumret är ej tillåtet.":
                                print("⚠️ Skylten är ej godkänd! Stoppar bildtagning.")
                                capture_running = False  # 🛑 Stoppar bildtagningen

                            elif result["status"] == "invalid" and result["message"] == "Kunde inte hitta registreringsnumret i databasen.":
                                print("⚠️ Skylten är ej godkänd! Stoppar bildtagning.")
                                capture_running = False  # 🛑 Stoppar bildtagningen
                        else:
                            print("❌ Servern kunde inte identifiera en skylt. Fortsätter bildtagning...")

                else:
                    print("⚠️ Ingen registreringsskylt identifierad, fortsätter skanna...")

            else:
                print("❌ Kunde inte hämta bild från IP-kameran")

        except Exception as e:
            print(f"⚠️ Fel: {e}")

        elapsed_time = time.time() - start_time
        sleep_time = max(0, 2 - elapsed_time)
        time.sleep(sleep_time)

    print("🛑 Bildtagningen har stoppats.")

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
