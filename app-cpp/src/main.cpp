#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h>

using namespace cv;
using namespace std;

const string SERVER_URL = "http://localhost:5000/detect";
const int DETECTION_DELAY = 30; // Frames de espera para no saturar

// imagen vía HTTP
void sendImageToBot(const Mat& frame) {
    CURL *curl;
    CURLcode res;
    vector<uchar> buf;
    imencode(".jpg", frame, buf);

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
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1000L); 

        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            cerr << "[ERROR HTTP] Fallo el envio: " << curl_easy_strerror(res) << endl;
        } else {
            cout << "[EXITO] Imagen enviada al servidor." << endl;
        }

        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }
}

int main() {
    curl_global_init(CURL_GLOBAL_ALL);

    // cargar Cascada LBP
    CascadeClassifier detector;
    if (!detector.load("cascade.xml")) {
        cerr << "[ERROR CRITICO] No se pudo cargar 'cascade.xml'. Revise la ruta." << endl;
        return -1;
    }
    cout << "[INFO] Modelo LBP cargado correctamente." << endl;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] No se pudo abrir la cámara web." << endl;
        return -1;
    }
    
    // Resolución baja para velocidad (LBP funciona muy bien en baja res)
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame, gray;
    vector<Rect> detections;
    int cooldownCounter = 0;

    cout << "Sistema de Vigilancia Iniciado (Modo LBP)" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray); // Importante para LBP

        detector.detectMultiScale(gray, detections, 1.05, 8, 0, Size(70, 140));

        for (const auto& rect : detections) {
            rectangle(frame, rect, Scalar(0, 255, 0), 2);
            putText(frame, "Persona", Point(rect.x, rect.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 1);
        }

        if (!detections.empty()) {
            if (cooldownCounter == 0) {
                sendImageToBot(frame);
                cooldownCounter = DETECTION_DELAY; 
            }
        }

        if (cooldownCounter > 0) cooldownCounter--;

        imshow("Camara Vigilancia LBP", frame);
        if (waitKey(1) == 'q') break;
    }

    curl_global_cleanup();
    return 0;
}