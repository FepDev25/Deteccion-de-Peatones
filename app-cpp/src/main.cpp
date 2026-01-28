#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h>
#include <thread>
#include <atomic>
#include <string>

using namespace cv;
using namespace std;

// config
const string SERVER_URL = "http://localhost:5000/detect";
const int DETECTION_DELAY = 30; // Frames de espera entre envíos

const double THRESHOLD_STANDING = 1.1; 
const double THRESHOLD_CROUCHING = 1.0;

// Bandera atómica para controlar estado de envío
atomic<bool> isBusy(false);

// Función de envío asíncrono (No bloquea el video)
void sendImageAsync(Mat frameToSend) {
    isBusy = true; 

    CURL *curl;
    CURLcode res;
    vector<uchar> buf;

    // Medir tiempo
    double tick = (double)getTickCount();

    // Codificar jpg
    imencode(".jpg", frameToSend, buf);

    curl = curl_easy_init();
    if(curl) {
        curl_mime *mime;
        curl_mimepart *part;
        mime = curl_mime_init(curl);

        part = curl_mime_addpart(mime);
        curl_mime_name(part, "image");
        curl_mime_data(part, (const char*)buf.data(), buf.size());
        curl_mime_filename(part, "capture.jpg");
        curl_mime_type(part, "image/jpeg");

        curl_easy_setopt(curl, CURLOPT_URL, SERVER_URL.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 5000L); // 5s timeout

        res = curl_easy_perform(curl);

        double time_sec = ((double)getTickCount() - tick) / getTickFrequency();

        if(res != CURLE_OK) {
            cerr << "[RED-ERROR] Fallo envio (" << time_sec << "s): " << curl_easy_strerror(res) << endl;
        } else {
            cout << "[RED-OK] Enviado en " << time_sec << "s" << endl;
        }

        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }

    isBusy = false; 
}

int main() {
    curl_global_init(CURL_GLOBAL_ALL);

    // modelos
    CascadeClassifier detectorStanding;
    CascadeClassifier detectorCrouching;

    if (!detectorStanding.load("cascade_standing.xml")) {
        cerr << "[ERROR CRITICO] Falta 'cascade_standing.xml'" << endl;
        return -1;
    }
    if (!detectorCrouching.load("cascade_crouching.xml")) {
        cerr << "[ERROR CRITICO] Falta 'cascade_crouching.xml'" << endl;
        return -1;
    }

    cout << "[INFO] Modelos cargados. Iniciando Filtro de Confianza..." << endl;

    // Configurar Cámara
    VideoCapture cap(0, CAP_V4L2);
    if (!cap.isOpened()) cap.open(0);
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);

    Mat frame, gray;
    // Vectores para almacenar resultados, niveles de rechazo y pesos (confianza)
    vector<Rect> detectionsStanding, detectionsCrouching;
    vector<int> rejectLevels;
    vector<double> levelWeights;
    
    int cooldownCounter = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray); 

        bool personDetected = false;

        // deteccion de pie
        detectionsStanding.clear(); rejectLevels.clear(); levelWeights.clear();
        
        // detectMultiScale con salida de pesos (true al final)
        detectorStanding.detectMultiScale(gray, detectionsStanding, rejectLevels, levelWeights, 
                                          1.1, 4, 0, Size(50, 100), Size(), true);

        for (size_t i = 0; i < detectionsStanding.size(); i++) {
            // FILTRO EXPERTO: Solo aceptamos si la confianza supera el umbral
            if (levelWeights[i] > THRESHOLD_STANDING) {
                Rect rect = detectionsStanding[i];
                rectangle(frame, rect, Scalar(0, 255, 0), 2); // Verde
                
                // Mostrar nivel de confianza en pantalla
                string label = "Pie: " + to_string(levelWeights[i]).substr(0, 3);
                putText(frame, label, Point(rect.x, rect.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1);
                
                personDetected = true;
            }
        }

        // deteccion agachado
        detectionsCrouching.clear(); rejectLevels.clear(); levelWeights.clear();
        
        detectorCrouching.detectMultiScale(gray, detectionsCrouching, rejectLevels, levelWeights, 
                                           1.1, 4, 0, Size(40, 40), Size(), true);

        for (size_t i = 0; i < detectionsCrouching.size(); i++) {
            if (levelWeights[i] > THRESHOLD_CROUCHING) {
                Rect rect = detectionsCrouching[i];
                rectangle(frame, rect, Scalar(0, 255, 255), 2); // Amarillo
                
                string label = "Agachado: " + to_string(levelWeights[i]).substr(0, 3);
                putText(frame, label, Point(rect.x, rect.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
                
                personDetected = true;
            }
        }

        if (personDetected) {
            if (cooldownCounter == 0 && !isBusy) {
                cout << "[ALERTA] Persona confirmada (Filtro Superado). Enviando..." << endl;
                thread t(sendImageAsync, frame.clone());
                t.detach(); 
                cooldownCounter = DETECTION_DELAY; 
            }
        }

        if (cooldownCounter > 0) cooldownCounter--;

        // Indicador visual de estado
        if (isBusy) circle(frame, Point(20, 20), 8, Scalar(0, 0, 255), -1);
        else circle(frame, Point(20, 20), 8, Scalar(0, 255, 0), -1);

        imshow("Vigilancia Experta LBP", frame);
        if (waitKey(1) == 'q') break;
    }

    curl_global_cleanup();
    return 0;
}