#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h>
#include <chrono>
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
        equalizeHist(gray, gray); 

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