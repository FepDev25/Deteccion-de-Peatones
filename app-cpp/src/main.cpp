#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h>
#include <chrono>
#include <string>

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

int main(int argc, char** argv) {
    curl_global_init(CURL_GLOBAL_ALL);

    // Determinar qué modelo usar
    string modelo_path = "cascade_standing.xml"; // Por defecto, personas paradas
    
    if (argc > 1) {
        string arg = argv[1];
        if (arg == "crouching" || arg == "agachadas") {
            modelo_path = "cascade_crouching.xml";
            cout << "[INFO] Modo: Detección de personas AGACHADAS/SENTADAS" << endl;
        } else if (arg == "standing" || arg == "paradas") {
            modelo_path = "cascade_standing.xml";
            cout << "[INFO] Modo: Detección de personas PARADAS" << endl;
        } else if (arg == "dual") {
            cout << "[INFO] Modo DUAL: Detectará ambos tipos" << endl;
        }
    } else {
        cout << "[INFO] Modo por defecto: Detección de personas PARADAS" << endl;
        cout << "[INFO] Uso: ./app_vigilante [standing|crouching|dual]" << endl;
    }

    // Cargar Cascada LBP
    CascadeClassifier detector;
    if (!detector.load(modelo_path)) {
        cerr << "[ERROR CRITICO] No se pudo cargar '" << modelo_path << "'. Revise la ruta." << endl;
        cerr << "[INFO] Asegúrate de ejecutar desde la carpeta 'build/'" << endl;
        return -1;
    }
    cout << "[✓] Modelo LBP cargado correctamente: " << modelo_path << endl;

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
    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();

    cout << "\n========================================" << endl;
    cout << "  Sistema de Vigilancia Iniciado (LBP)" << endl;
    cout << "========================================" << endl;
    cout << "Controles:" << endl;
    cout << "  [Q] - Salir" << endl;
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

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray); // Importante para LBP

        // Parámetros mejorados para reducir falsos positivos:
        // - scaleFactor: 1.1 (más grande = menos detecciones pero más precisas)
        // - minNeighbors: 12 (más alto = menos falsos positivos)
        // - minSize: 100x200 (personas más grandes, evita detecciones pequeñas)
        detector.detectMultiScale(gray, detections, 1.1, 12, 0, Size(100, 200));

        // Dibujar detecciones
        for (const auto& rect : detections) {
            rectangle(frame, rect, Scalar(0, 255, 0), 2);
            putText(frame, "Persona", Point(rect.x, rect.y - 5), 
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
        }

        // Mostrar info en pantalla
        putText(frame, "FPS: " + to_string(int(fps)), Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
        putText(frame, "Personas: " + to_string(detections.size()), Point(10, 65), 
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
        
        if (cooldownCounter > 0) {
            putText(frame, "Cooldown: " + to_string(cooldownCounter), Point(10, 100), 
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

        imshow("Camara Vigilancia LBP", frame);
        
        // Manejo de teclado
        int key = waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            cout << "\n[INFO] Finalizando programa..." << endl;
            break;
        } else if (key == 32) { // ESPACIO
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