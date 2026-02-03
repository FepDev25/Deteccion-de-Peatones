#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <curl/curl.h>
#include <chrono>
#include <string>
#include <set>

using namespace cv;
using namespace std;

const string SERVER_URL = "http://localhost:5000/detect";
const int DETECTION_DELAY = 90; // Frames de espera (solo si auto-envío activo)
bool AUTO_SEND_ENABLED = false; // Desactivado por defecto - usar ESPACIO para enviar

// VALIDACIÓN ROBUSTA: Distinguir personas reales de ventanas/muebles
// Personas tienen densidad de bordes moderada y gradientes verticales
bool isValidPerson(const Mat& roi) {
    if (roi.empty() || roi.cols < 50 || roi.rows < 50) return false;
    
    Mat grayRoi, edges;
    if (roi.channels() == 3) {
        cvtColor(roi, grayRoi, COLOR_BGR2GRAY);
    } else {
        grayRoi = roi.clone();
    }
    
    // Reducir ruido antes de detección de bordes
    GaussianBlur(grayRoi, grayRoi, Size(5, 5), 1.5);
    
    // Detectar bordes con Canny
    Canny(grayRoi, edges, 50, 150);
    
    // Calcular densidad de bordes
    int edgePixels = countNonZero(edges);
    float edgeDensity = (float)edgePixels / (roi.cols * roi.rows);
    
    // FILTRO 1: Densidad de bordes (MÁS PERMISIVO)
    // Ventanas: < 0.04 (solo marco)
    // Personas: 0.05 - 0.50 (contorno corporal, ropa, cabello)
    // Ruido: > 0.55 (demasiado ruidoso)
    if (edgeDensity < 0.04 || edgeDensity > 0.52) {
        return false;
    }
    
    // FILTRO 2: Gradientes verticales (estructura de persona) - MÁS PERMISIVO
    Mat sobelY;
    Sobel(grayRoi, sobelY, CV_32F, 0, 1, 3);
    Scalar mean, stddev;
    meanStdDev(abs(sobelY), mean, stddev);
    
    // Personas tienen variación vertical significativa (reducido de 12 a 8)
    if (stddev[0] < 8.0) {
        return false;
    }
    
    return true;
}

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
            // No mostrar error si el bot no está corriendo
        } else {
            cout << "[✓] Imagen enviada al bot" << endl;
        }

        curl_mime_free(mime);
        curl_easy_cleanup(curl);
    }
}

// Función para filtrar detecciones superpuestas (Non-Maximum Suppression)
void filterOverlappingDetections(vector<Rect>& detections, vector<string>& labels, float overlapThresh = 0.3) {
    if (detections.empty()) return;
    
    vector<Rect> picked;
    vector<string> pickedLabels;
    
    // Ordenar por área (más grande primero)
    vector<int> indices(detections.size());
    for (size_t i = 0; i < detections.size(); i++) {
        indices[i] = i;
    }
    
    sort(indices.begin(), indices.end(), [&detections](int i1, int i2) {
        return detections[i1].area() > detections[i2].area();
    });
    
    vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        picked.push_back(detections[idx]);
        pickedLabels.push_back(labels[idx]);
        
        // Suprimir detecciones superpuestas
        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            Rect intersection = detections[idx] & detections[idx2];
            float iou = (float)intersection.area() / (float)(detections[idx].area() + detections[idx2].area() - intersection.area());
            
            if (iou > overlapThresh) {
                suppressed[idx2] = true;
            }
        }
    }
    
    detections = picked;
    labels = pickedLabels;
}

int main(int argc, char** argv) {
    curl_global_init(CURL_GLOBAL_ALL);

    cout << "\n========================================" << endl;
    cout << "  Detector Multi-Postura LBP" << endl;
    cout << "  Técnica Clásica: Cascadas de Haar" << endl;
    cout << "========================================\n" << endl;

    // Cargar AMBOS detectores LBP
    CascadeClassifier detectorStanding, detectorCrouching;
    
    bool standingLoaded = detectorStanding.load("cascade_standing.xml");
    bool crouchingLoaded = detectorCrouching.load("cascade_crouching.xml");
    
    if (!standingLoaded && !crouchingLoaded) {
        cerr << "[✗] Error: No se pudo cargar ningún modelo" << endl;
        cerr << "[INFO] Asegúrate de estar en la carpeta build/" << endl;
        return -1;
    }
    
    if (standingLoaded) cout << "[✓] Modelo personas paradas cargado" << endl;
    if (crouchingLoaded) cout << "[✓] Modelo personas agachadas cargado" << endl;

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "[✗] Error: No se pudo abrir la cámara" << endl;
        return -1;
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cout << "[✓] Cámara inicializada: 640x480\n" << endl;

    Mat frame, gray;
    vector<Rect> detectionsStanding, detectionsCrouching, allDetections;
    vector<string> allLabels;
    int cooldownCounter = 0;
    int frameCount = 0;
    auto startTime = chrono::high_resolution_clock::now();

    cout << "========================================" << endl;
    cout << "  Sistema Iniciado" << endl;
    cout << "========================================" << endl;
    cout << "Controles:" << endl;
    cout << "  [Q/ESC] - Salir" << endl;
    cout << "  [ESPACIO] - Enviar frame a Telegram (manual)" << endl;
    cout << "  [B] - Activar/Desactivar auto-envío a Telegram" << endl;
    cout << "  [1] - Solo personas paradas" << endl;
    cout << "  [2] - Solo personas agachadas" << endl;
    cout << "  [3] - Ambos detectores (por defecto)" << endl;
    cout << "========================================" << endl;
    cout << "🔴 Bot DESACTIVADO por defecto" << endl;
    cout << "   Presiona [B] para activar o [ESPACIO] para envío manual" << endl;
    cout << "========================================\n" << endl;

    bool useStanding = true;
    bool useCrouching = true;

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

        allDetections.clear();
        allLabels.clear();

        // Detector 1: Personas PARADAS (ventana vertical)
        if (useStanding && standingLoaded) {
            detectionsStanding.clear();
            detectorStanding.detectMultiScale(
                gray, 
                detectionsStanding, 
                1.06,           // scaleFactor más fino (detecta más escalas)
                14,             // minNeighbors BALANCEADO (era 24)
                0, 
                Size(90, 180),  // minSize REDUCIDO para personas parciales
                Size(280, 560)  // maxSize aumentado
            );
            
            for (const auto& rect : detectionsStanding) {
                // Filtro 1: Tamaño mínimo REDUCIDO
                if (rect.width < 70 || rect.height < 140) continue;
                
                // Filtro 2: Aspect ratio vertical MÁS PERMISIVO
                float aspectRatio = (float)rect.height / (float)rect.width;
                if (aspectRatio < 1.5 || aspectRatio > 3.5) continue;
                
                // Filtro 3: VALIDACIÓN DE BORDES
                Rect safeRect(
                    max(0, rect.x), 
                    max(0, rect.y),
                    min(rect.width, gray.cols - rect.x),
                    min(rect.height, gray.rows - rect.y)
                );
                
                Mat roi = frame(safeRect);
                if (!isValidPerson(roi)) {
                    continue; // Rechazar si no tiene estructura de persona
                }
                
                allDetections.push_back(rect);
                allLabels.push_back("Parado");
            }
        }

        // Detector 2: Personas AGACHADAS/SENTADAS (ventana más cuadrada)
        if (useCrouching && crouchingLoaded) {
            detectionsCrouching.clear();
            detectorCrouching.detectMultiScale(
                gray, 
                detectionsCrouching, 
                1.08,           // scaleFactor BALANCEADO (era 1.18)
                16,             // minNeighbors RAZONABLE (era 35)
                0, 
                Size(80, 70),   // minSize REDUCIDO para detectar agachados reales
                Size(300, 280)  // maxSize aumentado
            );
            
            for (const auto& rect : detectionsCrouching) {
                // Filtro 1: Tamaño mínimo REDUCIDO
                if (rect.width < 60 || rect.height < 50) continue;
                
                // Filtro 2: Aspect ratio MÁS PERMISIVO para agachados
                float aspectRatio = (float)rect.height / (float)rect.width;
                if (aspectRatio < 0.7 || aspectRatio > 2.0) continue;
                
                // Filtro 3: VALIDACIÓN DE BORDES (elimina ventanas/marcos)
                Rect safeRect(
                    max(0, rect.x), 
                    max(0, rect.y),
                    min(rect.width, gray.cols - rect.x),
                    min(rect.height, gray.rows - rect.y)
                );
                
                Mat roi = frame(safeRect);
                if (!isValidPerson(roi)) {
                    continue; // Rechazar si no tiene estructura de persona
                }
                
                allDetections.push_back(rect);
                allLabels.push_back("Agachado");
            }
        }

        // Filtrar detecciones superpuestas (IoU más permisivo para detectar múltiples personas)
        filterOverlappingDetections(allDetections, allLabels, 0.45);

        // Dibujar detecciones con colores diferentes
        for (size_t i = 0; i < allDetections.size(); i++) {
            Rect r = allDetections[i];
            Scalar color = (allLabels[i] == "Parado") ? Scalar(0, 255, 0) : Scalar(0, 165, 255);
            
            rectangle(frame, r, color, 3);
            putText(frame, allLabels[i], Point(r.x, r.y - 10), 
                   FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }

        // Información en pantalla
        putText(frame, "FPS: " + to_string(int(fps)), Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
        putText(frame, "Personas: " + to_string(allDetections.size()), Point(10, 65), 
               FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
        
        string mode = (useStanding && useCrouching) ? "Dual" : (useStanding ? "Paradas" : "Agachadas");
        putText(frame, "Modo: " + mode, Point(10, 100), 
               FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 255), 2);
        
        if (cooldownCounter > 0) {
            putText(frame, "Cooldown: " + to_string(cooldownCounter), Point(10, 135), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 165, 255), 2);
        }

        // Enviar al bot SOLO si auto-envío está activado Y hay detección
        if (AUTO_SEND_ENABLED && !allDetections.empty()) {
            if (cooldownCounter == 0) {
                sendImageToBot(frame);
                cooldownCounter = DETECTION_DELAY; 
            }
        }

        if (cooldownCounter > 0) cooldownCounter--;

        imshow("Detector Multi-Postura - LBP", frame);
        
        // Manejo de teclado
        int key = waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
            cout << "\n[INFO] Finalizando..." << endl;
            break;
        } else if (key == 32) {
            cout << "[INFO] Envío manual..." << endl;
            sendImageToBot(frame);
            cooldownCounter = DETECTION_DELAY;
        } else if (key == '1') {
            useStanding = true;
            useCrouching = false;
            cout << "[MODO] Solo personas paradas" << endl;
        } else if (key == '2') {
            useStanding = false;
            useCrouching = true;
            cout << "[MODO] Solo personas agachadas" << endl;
        } else if (key == '3') {
            useStanding = true;
            useCrouching = true;
            cout << "[MODO] Detector dual (todos)" << endl;
        } else if (key == 'b' || key == 'B') {
            AUTO_SEND_ENABLED = !AUTO_SEND_ENABLED;
            cout << "[BOT] Auto-envío: " << (AUTO_SEND_ENABLED ? "ACTIVADO ✅" : "DESACTIVADO 🔴") << endl;
            cout << "      (Usa ESPACIO para envío manual)" << endl;
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
