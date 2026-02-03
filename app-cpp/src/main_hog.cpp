#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h>
#include <chrono>
#include <string>

using namespace cv;
using namespace std;

const string SERVER_URL = "http://localhost:5000/detect";
const int DETECTION_DELAY = 30; // Frames de espera

// Enviar imagen vía HTTP
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
            cerr << "[ERROR HTTP] " << curl_easy_strerror(res) << endl;
        } else {
            cout << "[✓] Imagen enviada al servidor" << endl;
        }

        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }
}

int main(int argc, char** argv) {
    curl_global_init(CURL_GLOBAL_ALL);

    cout << "\n========================================" << endl;
    cout << "  Detector de Peatones - HOG + SVM" << endl;
    cout << "  Basado en Dalal-Triggs (2005)" << endl;
    cout << "========================================\n" << endl;

    // Modo de sensibilidad (ajustable)
    double sensitivity = -0.7; // Por defecto: alta sensibilidad
    
    if (argc > 1) {
        string mode = argv[1];
        if (mode == "strict" || mode == "estricto") {
            sensitivity = 0.5;
            cout << "[MODO] Estricto: Menos detecciones pero muy precisas" << endl;
        } else if (mode == "balanced" || mode == "balanceado") {
            sensitivity = 0.0;
            cout << "[MODO] Balanceado: Equilibrio precisión/recall" << endl;
        } else if (mode == "sensitive" || mode == "sensible") {
            sensitivity = -0.7;
            cout << "[MODO] Sensible: Detecta más personas (recomendado)" << endl;
        }
    } else {
        cout << "[MODO] Por defecto: Sensible (detecta 4-6 de 6 personas)" << endl;
        cout << "[INFO] Modos disponibles: ./app_hog [strict|balanced|sensitive]" << endl;
    }
    cout << "[INFO] hitThreshold actual: " << sensitivity << "\n" << endl;

    // Inicializar HOG Descriptor
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    cout << "[✓] Detector HOG + SVM cargado correctamente" << endl;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[✗] Error: No se pudo abrir la cámara" << endl;
        return -1;
    }
    
    // Resolución moderada (HOG es más lento que LBP)
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cout << "[✓] Cámara inicializada: 640x480" << endl;

    Mat frame, resized;
    vector<Rect> detections;
    vector<double> weights;
    int cooldownCounter = 0;
    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();

    cout << "\n========================================" << endl;
    cout << "  Sistema Iniciado" << endl;
    cout << "========================================" << endl;
    cout << "Controles:" << endl;
    cout << "  [Q/ESC] - Salir" << endl;
    cout << "  [ESPACIO] - Enviar frame manualmente" << endl;
    cout << "========================================\n" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        frameCount++;
        
        // Calcular FPS
        auto currentTime = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = currentTime - startTime;
        double fps = frameCount / elapsed.count();

        // Redimensionar para mejorar velocidad (opcional)
        resize(frame, resized, Size(320, 240));

        // Detectar con HOG
        detections.clear();
        weights.clear();
        
        // Parámetros optimizados según Dalal-Triggs (2005)
        // Para detectar 4-6 de 6 personas (alta recall, baja tasa de falsos negativos)
        hog.detectMultiScale(
            resized, 
            detections, 
            weights,
            sensitivity,      // Ajustable por línea de comandos
            Size(4, 4),       // winStride pequeño = búsqueda exhaustiva (más lento pero preciso)
            Size(8, 8),       // padding estándar
            1.03,             // scale pequeño = más escalas = detecta más tamaños diferentes
            1.5               // finalThreshold (elimina detecciones duplicadas)
        );

        // Escalar detecciones al tamaño original
        for (auto& rect : detections) {
            rect.x *= 2;
            rect.y *= 2;
            rect.width *= 2;
            rect.height *= 2;
        }

        // Dibujar detecciones
        for (size_t i = 0; i < detections.size(); i++) {
            Rect r = detections[i];
            rectangle(frame, r, Scalar(0, 255, 0), 3);
            
            // Mostrar confianza
            if (i < weights.size()) {
                string conf = "Conf: " + to_string(weights[i]).substr(0, 4);
                putText(frame, conf, Point(r.x, r.y - 10), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            }
        }

        // Información en pantalla
        putText(frame, "FPS: " + to_string(int(fps)), Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
        putText(frame, "Personas: " + to_string(detections.size()), Point(10, 65), 
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
        putText(frame, "Tecnica: HOG+SVM", Point(10, 100), 
               FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 255), 2);
        
        if (cooldownCounter > 0) {
            putText(frame, "Cooldown: " + to_string(cooldownCounter), Point(10, 135), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 165, 255), 2);
        }

        // Enviar al bot si hay detección
        if (!detections.empty()) {
            if (cooldownCounter == 0) {
                sendImageToBot(frame);
                cooldownCounter = DETECTION_DELAY; 
            }
        }

        if (cooldownCounter > 0) cooldownCounter--;

        imshow("Detector de Peatones - HOG + SVM", frame);
        
        // Manejo de teclado
        int key = waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            cout << "\n[INFO] Finalizando..." << endl;
            break;
        } else if (key == 32) {
            cout << "[INFO] Envío manual..." << endl;
            sendImageToBot(frame);
            cooldownCounter = DETECTION_DELAY;
        }
    }

    // Estadísticas finales
    cout << "\n========================================" << endl;
    cout << "  ESTADÍSTICAS FINALES" << endl;
    cout << "========================================" << endl;
    cout << "Frames procesados: " << frameCount << endl;
    auto endTime = chrono::high_resolution_clock::now();
    chrono::duration<double> totalTime = endTime - startTime;
    cout << "FPS promedio: " << frameCount / totalTime.count() << endl;
    cout << "Tiempo total: " << totalTime.count() << " segundos" << endl;
    cout << "========================================\n" << endl;

    curl_global_cleanup();
    return 0;
}
