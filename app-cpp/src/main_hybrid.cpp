#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>

using namespace cv;
using namespace std;

// ==================== CONFIGURACIÓN ====================
const string CASCADE_CROUCHING = "cascade_crouching.xml";
const string BOT_URL = "http://localhost:5000/detect";
const bool AUTO_SEND_ENABLED = true;  // AUTO-ENVÍO ACTIVADO
const int COOLDOWN_FRAMES = 60;        // Cada ~2 segundos (a 30 FPS)

// ==================== VALIDACIÓN DE ROI ====================
bool isValidPerson(const Mat& roi) {
    // Validación inteligente para LBP (eliminar ventanas sin perder personas)
    Mat gray_roi;
    if (roi.channels() == 3) {
        cvtColor(roi, gray_roi, COLOR_BGR2GRAY);
    } else {
        gray_roi = roi.clone();
    }

    // 1. Varianza de color - ventanas son muy uniformes, personas tienen textura
    Scalar mean_color, stddev_color;
    meanStdDev(gray_roi, mean_color, stddev_color);
    if (stddev_color[0] < 17.0) {  // Aumentado para filtrar sillas uniformes
        return false; // Demasiado uniforme (ventana/pared/silla)
    }

    // 2. Canny edge detection - densidad de bordes
    Mat edges;
    Canny(gray_roi, edges, 50, 150);
    double edgeDensity = countNonZero(edges) / (double)(edges.rows * edges.cols);
    
    // Ventanas/sillas tienen baja densidad O demasiada (marcos metálicos)
    if (edgeDensity < 0.08 || edgeDensity > 0.40) {  // Reducido máximo para filtrar sillas
        return false;
    }

    // 3. Sobel gradients - verificar estructura humana
    Mat grad_x, grad_y;
    Sobel(gray_roi, grad_x, CV_16S, 1, 0, 3);
    Sobel(gray_roi, grad_y, CV_16S, 0, 1, 3);
    
    Mat abs_grad_x, abs_grad_y;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    Scalar mean_x, stddev_x, mean_y, stddev_y;
    meanStdDev(abs_grad_x, mean_x, stddev_x);
    meanStdDev(abs_grad_y, mean_y, stddev_y);
    
    // Personas tienen gradientes balanceados en ambas direcciones
    if (stddev_x[0] < 9.0 || stddev_y[0] < 9.0) {  // Aumentado
        return false;
    }

    // 4. Análisis de textura LBP - contar transiciones de píxeles
    int transitions = 0;
    for (int i = 1; i < gray_roi.rows - 1; i++) {
        for (int j = 1; j < gray_roi.cols - 1; j++) {
            int center = gray_roi.at<uchar>(i, j);
            int diff = 0;
            diff += abs(center - gray_roi.at<uchar>(i-1, j)) > 10 ? 1 : 0;
            diff += abs(center - gray_roi.at<uchar>(i+1, j)) > 10 ? 1 : 0;
            diff += abs(center - gray_roi.at<uchar>(i, j-1)) > 10 ? 1 : 0;
            diff += abs(center - gray_roi.at<uchar>(i, j+1)) > 10 ? 1 : 0;
            if (diff >= 2) transitions++;
        }
    }
    double transitionRatio = transitions / (double)(gray_roi.rows * gray_roi.cols);
    if (transitionRatio < 0.15) {  // Aumentado para filtrar sillas
        return false; // Muy poca textura (ventana lisa o silla)
    }

    // 5. Aspect ratio adicional (evitar detecciones muy anchas)
    double aspect = (double)roi.cols / roi.rows;
    if (aspect > 2.3) {
        return false;
    }

    // 6. Detección de tono de piel (filtro adicional)
    Mat hsv_roi;
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    Mat skin_mask;
    inRange(hsv_roi, Scalar(0, 20, 60), Scalar(20, 255, 255), skin_mask);
    double skinRatio = countNonZero(skin_mask) / (double)(skin_mask.rows * skin_mask.cols);
    if (skinRatio < 0.012) {  // Al menos 1.2% de píxeles con tono piel
        return false;
    }

    return true;
}

// ==================== FILTRO DE DETECCIONES SUPERPUESTAS ====================
void filterOverlappingDetections(vector<Rect>& detections, vector<string>& labels, 
                                  vector<double>& confidences, double iouThreshold = 0.45) {
    if (detections.empty()) return;

    vector<int> indices;
    for (size_t i = 0; i < detections.size(); i++) {
        indices.push_back(i);
    }

    // Ordenar por confianza (mayor a menor)
    sort(indices.begin(), indices.end(), [&confidences](int a, int b) {
        return confidences[a] > confidences[b];
    });

    vector<bool> suppressed(detections.size(), false);
    vector<Rect> filtered_detections;
    vector<string> filtered_labels;
    vector<double> filtered_confidences;

    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        filtered_detections.push_back(detections[idx]);
        filtered_labels.push_back(labels[idx]);
        filtered_confidences.push_back(confidences[idx]);

        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;

            Rect intersection = detections[idx] & detections[idx2];
            double intersectionArea = intersection.area();
            double unionArea = detections[idx].area() + detections[idx2].area() - intersectionArea;
            double iou = intersectionArea / unionArea;

            if (iou > iouThreshold) {
                suppressed[idx2] = true;
            }
        }
    }

    detections = filtered_detections;
    labels = filtered_labels;
    confidences = filtered_confidences;
}

// ==================== ENVÍO AL BOT ====================
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

bool sendImageToBot(const Mat& frame) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    vector<uchar> buf;
    imencode(".jpg", frame, buf);

    struct curl_httppost* formpost = NULL;
    struct curl_httppost* lastptr = NULL;
    struct curl_slist* headerlist = NULL;

    curl_formadd(&formpost, &lastptr,
                 CURLFORM_COPYNAME, "image",
                 CURLFORM_BUFFER, "detection.jpg",
                 CURLFORM_BUFFERPTR, buf.data(),
                 CURLFORM_BUFFERLENGTH, buf.size(),
                 CURLFORM_CONTENTTYPE, "image/jpeg",
                 CURLFORM_END);

    string response_string;
    curl_easy_setopt(curl, CURLOPT_URL, BOT_URL.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

    CURLcode res = curl_easy_perform(curl);
    bool success = (res == CURLE_OK);

    curl_easy_cleanup(curl);
    curl_formfree(formpost);

    return success;
}

// ==================== MAIN ====================
int main() {
    // Inicializar cámara
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: No se pudo abrir la camara" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // Cargar detector HOG (pre-entrenado)
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // Cargar cascade LBP para agachados
    CascadeClassifier cascade_crouching;
    if (!cascade_crouching.load(CASCADE_CROUCHING)) {
        cerr << "Error: No se pudo cargar " << CASCADE_CROUCHING << endl;
        return -1;
    }

    cout << "\n========================================" << endl;
    cout << "  DETECTOR HIBRIDO HOG+LBP" << endl;
    cout << "========================================" << endl;
    cout << "HOG: Personas de pie (alta precision)" << endl;
    cout << "LBP: Personas agachadas/acostadas" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Tecla 1: Modo HOG (solo de pie)" << endl;
    cout << "Tecla 2: Modo LBP (solo agachados)" << endl;
    cout << "Tecla 3: Modo HIBRIDO (ambos)" << endl;
    cout << "Tecla B: Activar/Desactivar bot" << endl;
    cout << "Tecla ESPACIO: Enviar deteccion manual" << endl;
    cout << "Tecla Q: Salir" << endl;
    cout << "========================================\n" << endl;

    int mode = 3; // Por defecto: híbrido
    bool botEnabled = true;
    int frameCounter = 0;
    int cooldown = 0;
    
    // FPS calculation
    time_t start, end;
    double fps = 0;
    int frameCount = 0;
    time(&start);

    namedWindow("Detector Hibrido", WINDOW_NORMAL);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat display = frame.clone();
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Rect> all_detections;
        vector<string> all_labels;
        vector<double> all_confidences;

        // ============ MODO 1 o 3: HOG para personas de pie ============
        if (mode == 1 || mode == 3) {
            vector<Rect> hog_detections;
            vector<double> weights;
            
            // Reducir tamaño para mejor performance
            Mat resized;
            resize(frame, resized, Size(320, 240));
            
            // Detectar con HOG (hitThreshold = 0.0 balanceado)
            hog.detectMultiScale(resized, hog_detections, weights, 
                                 0.0,           // hitThreshold
                                 Size(4, 4),    // winStride
                                 Size(8, 8),    // padding
                                 1.05,          // scale
                                 2.0);          // finalThreshold

            // Escalar detecciones de vuelta a tamaño original
            for (size_t i = 0; i < hog_detections.size(); i++) {
                Rect r = hog_detections[i];
                r.x = (int)(r.x * 2.0);
                r.y = (int)(r.y * 2.0);
                r.width = (int)(r.width * 2.0);
                r.height = (int)(r.height * 2.0);

                // Filtro de aspect ratio (personas de pie)
                double aspect = (double)r.height / r.width;
                if (aspect >= 1.5 && aspect <= 3.0) {
                    all_detections.push_back(r);
                    all_labels.push_back("Peaton detectado");
                    all_confidences.push_back(weights[i]);
                }
            }
        }

        // ============ MODO 2 o 3: LBP para personas agachadas ============
        if (mode == 2 || mode == 3) {
            vector<Rect> lbp_detections;
            
            // Parámetros BALANCEADOS: detectar agachados sin falsos positivos
            cascade_crouching.detectMultiScale(
                gray, 
                lbp_detections,
                1.05,            // scaleFactor (más escalas)
                18,              // minNeighbors BALANCEADO
                CASCADE_SCALE_IMAGE,
                Size(75, 65),    // minSize (personas agachadas pequeñas)
                Size(320, 300)   // maxSize (personas sentadas grandes)
            );

            for (Rect r : lbp_detections) {
                // Filtro de aspect ratio (agachados/sentados/parciales)
                double aspect = (double)r.width / r.height;
                if (aspect < 0.6 || aspect > 2.3) continue;

                // FILTRAR SI YA HAY DETECCIÓN HOG CERCANA (evitar duplicados)
                bool overlapWithHOG = false;
                for (size_t i = 0; i < all_detections.size(); i++) {
                    if (all_labels[i] == "Peaton detectado") {
                        Rect intersection = r & all_detections[i];
                        double iou = intersection.area() / (double)r.area();
                        if (iou > 0.15) {  // Solo rechazar si overlap >15%
                            overlapWithHOG = true;
                            break;
                        }
                    }
                }
                if (overlapWithHOG) continue;

                // EXPANDIR BOUNDING BOX ASIMÉTRICAMENTE (compensar detección parcial LBP)
                int expandX = (int)(r.width * 0.25);       // 25% más ancho (brazos/hombros)
                int expandTop = (int)(r.height * 0.20);    // 20% hacia arriba (cabeza completa)
                int expandBottom = (int)(r.height * 0.60); // 60% hacia abajo (PIERNAS COMPLETAS)
                
                r.x = max(0, r.x - expandX);
                r.y = max(0, r.y - expandTop);
                r.width = min(frame.cols - r.x, r.width + 2 * expandX);
                r.height = min(frame.rows - r.y, r.height + expandTop + expandBottom);

                // Validación inteligente de ROI (filtra ventanas automáticamente)
                Mat roi = frame(r);
                if (!isValidPerson(roi)) continue;

                all_detections.push_back(r);
                all_labels.push_back("Peaton detectado");
                all_confidences.push_back(1.0); // LBP no da scores
            }
        }

        // Filtrar detecciones superpuestas (NMS)
        filterOverlappingDetections(all_detections, all_labels, all_confidences, 0.4);

        // Dibujar detecciones
        for (size_t i = 0; i < all_detections.size(); i++) {
            Scalar color = (all_labels[i] == "Peaton detectado") ? Scalar(0, 255, 0) : Scalar(255, 0, 255);
            rectangle(display, all_detections[i], color, 2);
            
            string label = all_labels[i];
            int baseline;
            Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            Point textOrg(all_detections[i].x, all_detections[i].y - 10);
            
            rectangle(display, textOrg + Point(0, baseline), 
                     textOrg + Point(textSize.width, -textSize.height), 
                     color, FILLED);
            putText(display, label, textOrg, FONT_HERSHEY_SIMPLEX, 0.6, 
                   Scalar(0, 0, 0), 2);
        }

        // Calcular FPS
        frameCount++;
        time(&end);
        double sec = difftime(end, start);
        if (sec >= 1.0) {
            fps = frameCount / sec;
            frameCount = 0;
            time(&start);
        }

        // Mostrar información
        string modeStr = (mode == 1) ? "HOG" : (mode == 2) ? "LBP" : "HIBRIDO";
        putText(display, "Modo: " + modeStr, Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        putText(display, "FPS: " + to_string((int)fps), Point(10, 60), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        putText(display, "Detecciones: " + to_string(all_detections.size()), 
               Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        putText(display, botEnabled ? "Bot: ON" : "Bot: OFF", Point(10, 120), 
               FONT_HERSHEY_SIMPLEX, 0.7, 
               botEnabled ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);

        imshow("Detector Hibrido", display);

        // Auto-envío al bot cuando hay detecciones
        if (AUTO_SEND_ENABLED && botEnabled && !all_detections.empty() && cooldown == 0) {
            cout << "[AUTO] Enviando deteccion al bot..." << endl;
            if (sendImageToBot(display)) {
                cout << "[AUTO] Imagen enviada exitosamente! (" << all_detections.size() << " personas)" << endl;
                cooldown = COOLDOWN_FRAMES;
            }
        }

        // Gestión de cooldown
        if (cooldown > 0) cooldown--;

        // Control de teclado
        int key = waitKey(1);
        if (key == 'q' || key == 'Q') break;
        if (key == '1') { mode = 1; cout << "Modo: HOG (solo de pie)" << endl; }
        if (key == '2') { mode = 2; cout << "Modo: LBP (solo agachados)" << endl; }
        if (key == '3') { mode = 3; cout << "Modo: HIBRIDO (ambos)" << endl; }
        if (key == 'b' || key == 'B') {
            botEnabled = !botEnabled;
            cout << (botEnabled ? "Bot ACTIVADO" : "Bot DESACTIVADO") << endl;
        }
        if (key == ' ') { // ESPACIO
            if (botEnabled && !all_detections.empty() && cooldown == 0) {
                cout << "Enviando deteccion al bot..." << endl;
                if (sendImageToBot(display)) {
                    cout << "Imagen enviada exitosamente!" << endl;
                    cooldown = COOLDOWN_FRAMES;
                } else {
                    cout << "Error al enviar imagen" << endl;
                }
            }
        }

        frameCounter++;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}