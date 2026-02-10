/**
 * Sistema de Vigilancia Híbrido - Módulo C++ (MOG2)
 * Detección de movimiento clásica como trigger para IA en Python
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h>
#include <thread>
#include <atomic>

using namespace cv;
using namespace std;

// Configuración
const string SERVER_URL = "http://localhost:5000/detect";
const int DETECTION_DELAY = 30;  // Cooldown en frames
const int MIN_AREA = 4000;       // Área mínima para detectar (px²)

atomic<bool> isBusy(false);  // Sincronización con servidor

// Envío HTTP asíncrono al servidor Python
void sendImageAsync(Mat frameToSend) {
    isBusy = true;

    CURL *curl;
    CURLcode res;
    vector<uchar> buf;
    double tick = (double)getTickCount();

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
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 5000L);

        res = curl_easy_perform(curl);
        double time_sec = ((double)getTickCount() - tick) / getTickFrequency();

        if(res != CURLE_OK) {
            cerr << "[RED-ERROR] Fallo envio (" << time_sec << "s): " << curl_easy_strerror(res) << endl;
        } else {
            cout << "[RED-OK] Enviado a Python en " << time_sec << "s" << endl;
        }

        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }

    isBusy = false;
}

int main() {
    curl_global_init(CURL_GLOBAL_ALL);

    // Inicializar cámara (preferir V4L2 en Linux)
    VideoCapture cap(0, CAP_V4L2);
    if (!cap.isOpened()) {
        cout << "[WARN] Backend V4L2 fallo, usando default..." << endl;
        cap.open(0);
    }

    if (!cap.isOpened()) {
        cerr << "[ERROR CRITICO] No se puede abrir la camara." << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 30);

    Mat frame, mask;

    // MOG2: history=500, threshold=16, detectShadows=true
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2(500, 16, true);

    int cooldownCounter = 0;
    cout << "=== VIGILANCIA INICIADA: MODO MOVIMIENTO (MOG2) ===" << endl;
    cout << "Esperando movimiento..." << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Sustracción de fondo
        pBackSub->apply(frame, mask);

        // Limpieza morfológica
        erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
        dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

        // Buscar contornos
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool significantMotion = false;
        Rect bigBox;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);

            if (area > MIN_AREA) {
                Rect rect = boundingRect(contours[i]);
                rectangle(frame, rect, Scalar(0, 0, 255), 2);
                putText(frame, "MOVIMIENTO", Point(rect.x, rect.y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,0,255), 2);
                significantMotion = true;
            }
        }

        // Trigger: enviar a IA si hay movimiento y no estamos en cooldown
        if (significantMotion && cooldownCounter == 0 && !isBusy) {
            cout << "[ALERTA] Movimiento detectado. Enviando a IA..." << endl;
            thread t(sendImageAsync, frame.clone());
            t.detach();
            cooldownCounter = DETECTION_DELAY;
        }

        if (cooldownCounter > 0) cooldownCounter--;

        // Indicador visual de estado
        if (isBusy) {
            circle(frame, Point(30, 30), 10, Scalar(0, 0, 255), -1);
            putText(frame, "IA Procesando...", Point(50, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255), 1);
        } else {
            circle(frame, Point(30, 30), 10, Scalar(0, 255, 0), -1);
        }

        imshow("Vigilancia Hibrida", frame);
        // imshow("Mascara (Debug)", mask);

        if (waitKey(1) == 'q') break;
    }

    curl_global_cleanup();
    return 0;
}