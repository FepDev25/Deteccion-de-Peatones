#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h> // Librería para HTTP

using namespace cv;
using namespace std;

// URL del Servidor Python
const string SERVER_URL = "http://localhost:5000/detect";
const int DETECTION_DELAY = 30; // Frames de espera para no saturar

// enviar la imagen vía HTTP POST
void sendImageToBot(const Mat& frame) {
    CURL *curl;
    CURLcode res;

    // codificar imagen a buffer JPG en memoria
    vector<uchar> buf;
    imencode(".jpg", frame, buf);

    // iniciar cURL
    curl = curl_easy_init();
    if(curl) {
        // Estructuras para el formulario multipart
        curl_mime *mime;
        curl_mimepart *part;

        mime = curl_mime_init(curl);

        // añadir el campo "image" con los bytes del buffer
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "image");
        curl_mime_data(part, (const char*)buf.data(), buf.size());
        curl_mime_filename(part, "capture.jpg"); // Nombre ficticio para el servidor
        curl_mime_type(part, "image/jpeg");

        // configurar la petición
        curl_easy_setopt(curl, CURLOPT_URL, SERVER_URL.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        
        // timeout para no colgar la cámara si el servidor no responde
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1000L); 

        // ejecutar envío
        cout << "[RED] Enviando imagen a " << SERVER_URL << "..." << endl;
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            cerr << "[ERROR HTTP] Fallo el envio: " << curl_easy_strerror(res) << endl;
            cerr << "       (¿Esta corriendo el servidor Python?)" << endl;
        } else {
            cout << "[EXITO] Imagen recibida por el servidor." << endl;
        }

        // limpieza
        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }
}

int main() {
    // inicializar entorno global de curl
    curl_global_init(CURL_GLOBAL_ALL);

    // configurar Detector HOG
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[ERROR] No se pudo abrir la cámara web." << endl;
        return -1;
    }
    
    // bajar resolución para mejorar rendimiento de red y FPS
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame, gray;
    vector<Rect> detections;
    int cooldownCounter = 0;

    cout << "Cliente C++" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        hog.detectMultiScale(gray, detections, 0, Size(8, 8), Size(32, 32), 1.05, 2);

        for (const auto& rect : detections) {
            rectangle(frame, rect, Scalar(0, 255, 0), 2);
        }

        // lógica de envío
        if (!detections.empty()) {
            if (cooldownCounter == 0) {
                // envío a Python
                sendImageToBot(frame);
                cooldownCounter = DETECTION_DELAY; 
            }
        }

        if (cooldownCounter > 0) cooldownCounter--;

        imshow("Cliente C++", frame);
        if (waitKey(1) == 'q') break;
    }

    // limpieza global
    curl_global_cleanup();
    return 0;
}