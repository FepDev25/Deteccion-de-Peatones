#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>   // HOG GPU
#include <opencv2/cudaimgproc.hpp>     // Color conversions GPU
#include <opencv2/cudawarping.hpp>     // Resize GPU
#include <curl/curl.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;

// ==================== CONFIGURACIÓN ====================
const string CASCADE_CROUCHING = "cascade_crouching.xml";
const string BOT_URL = "http://localhost:5000/detect";
const bool AUTO_SEND_ENABLED = true;
const int COOLDOWN_FRAMES = 60;

// ==================== VALIDACIÓN DE ROI ====================
bool isValidPerson(const Mat& roi) {
    Mat gray_roi;
    if (roi.channels() == 3) {
        cvtColor(roi, gray_roi, COLOR_BGR2GRAY);
    } else {
        gray_roi = roi.clone();
    }

    // 1. Varianza de color - BALANCEADO
    Scalar mean_color, stddev_color;
    meanStdDev(gray_roi, mean_color, stddev_color);
    if (stddev_color[0] < 18.0) {  // Balanceado: filtra uniformes pero permite ropa lisa
        return false;
    }

    // 2. Color de piel (HSV) - BALANCEADO
    Mat hsv_roi;
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    Mat skin_mask;
    inRange(hsv_roi, Scalar(0, 18, 65), Scalar(22, 255, 255), skin_mask);
    double skinRatio = countNonZero(skin_mask) / (double)(skin_mask.rows * skin_mask.cols);
    if (skinRatio < 0.015) {  // Al menos 1.5%
        return false;
    }

    // 3. Canny - BALANCEADO
    Mat edges;
    Canny(gray_roi, edges, 45, 135);
    double edgeDensity = countNonZero(edges) / (double)(edges.rows * edges.cols);
    if (edgeDensity < 0.09 || edgeDensity > 0.45) {  // Rango ajustado
        return false;
    }

    // 4. Sobel gradients - BALANCEADO
    Mat grad_x, grad_y;
    Sobel(gray_roi, grad_x, CV_16S, 1, 0, 3);
    Sobel(gray_roi, grad_y, CV_16S, 0, 1, 3);
    
    Mat abs_grad_x, abs_grad_y;
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    
    Scalar mean_x, stddev_x, mean_y, stddev_y;
    meanStdDev(abs_grad_x, mean_x, stddev_x);
    meanStdDev(abs_grad_y, mean_y, stddev_y);
    
    if (stddev_x[0] < 10.0 || stddev_y[0] < 10.0) {  // Balanceado
        return false;
    }

    // 5. Verificar área mínima razonable
    if (roi.rows < 40 || roi.cols < 30) {
        return false;  // Demasiado pequeño para ser una persona
    }

    return true;
}

// ==================== FILTRO NMS ====================
void filterOverlappingDetections(vector<Rect>& detections, vector<string>& labels, 
                                  vector<double>& confidences, double iouThreshold = 0.45) {
    if (detections.empty()) return;

    vector<int> indices;
    for (size_t i = 0; i < detections.size(); i++) {
        indices.push_back(i);
    }

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
    // Verificar soporte CUDA
    int deviceCount = cuda::getCudaEnabledDeviceCount();
    if (deviceCount == 0) {
        cerr << "❌ Error: No se detectó GPU CUDA compatible" << endl;
        cerr << "   Ejecuta la versión CPU: ./app_hybrid" << endl;
        return -1;
    }

    cout << "\n========================================" << endl;
    cout << "  DETECTOR HIBRIDO HOG+LBP (GPU)" << endl;
    cout << "  GPU Acelerada con CUDA" << endl;
    cout << "========================================" << endl;
    
    cuda::printShortCudaDeviceInfo(cuda::getDevice());
    cout << "----------------------------------------" << endl;
    cout << "HOG GPU: Personas de pie (ultra-rapido)" << endl;
    cout << "LBP CPU: Personas agachadas" << endl;
    cout << "========================================\n" << endl;

    // Inicializar cámara
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: No se pudo abrir la camara" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // Cargar HOG GPU (pre-entrenado)
    Ptr<cuda::HOG> hog_gpu = cuda::HOG::create(
        Size(64, 128),  // winSize (estándar Dalal-Triggs)
        Size(16, 16),   // blockSize
        Size(8, 8),     // blockStride
        Size(8, 8),     // cellSize
        9               // nbins
    );
    hog_gpu->setSVMDetector(hog_gpu->getDefaultPeopleDetector());
    hog_gpu->setNumLevels(13);       // Niveles de escala
    hog_gpu->setHitThreshold(0.5);   // Más estricto para reducir falsos positivos
    hog_gpu->setWinStride(Size(8, 8));  // DEBE ser múltiplo de blockStride
    hog_gpu->setScaleFactor(1.05);
    hog_gpu->setGroupThreshold(0);   // DEBE ser 0 cuando usamos weights (OBLIGATORIO)

    cout << "[GPU] ✓ HOG GPU inicializado" << endl;

    // Cargar cascade LBP para agachados (CPU)
    CascadeClassifier cascade_crouching;
    if (!cascade_crouching.load(CASCADE_CROUCHING)) {
        cerr << "Error: No se pudo cargar " << CASCADE_CROUCHING << endl;
        return -1;
    }
    cout << "[CPU] ✓ LBP cascade cargado" << endl;

    cout << "\n========================================" << endl;
    cout << "Controles:" << endl;
    cout << "  Tecla 1: Modo HOG GPU (solo de pie)" << endl;
    cout << "  Tecla 2: Modo LBP CPU (solo agachados)" << endl;
    cout << "  Tecla 3: Modo HIBRIDO (ambos)" << endl;
    cout << "  Tecla B: Activar/Desactivar bot" << endl;
    cout << "  Tecla ESPACIO: Enviar manual" << endl;
    cout << "  Tecla Q: Salir" << endl;
    cout << "========================================\n" << endl;

    int mode = 3;
    bool botEnabled = true;
    int frameCounter = 0;
    int cooldown = 0;
    
    // Métricas de rendimiento
    double total_gpu_time = 0;
    double total_cpu_time = 0;
    int frame_count = 0;

    // Buffers GPU
    cuda::GpuMat d_frame, d_gray, d_resized;

    namedWindow("Detector Hibrido GPU", WINDOW_NORMAL);

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        auto frame_start = chrono::high_resolution_clock::now();

        Mat display = frame.clone();
        vector<Rect> all_detections;
        vector<string> all_labels;
        vector<double> all_confidences;

        // ============ MODO 1 o 3: HOG GPU para personas de pie ============
        if (mode == 1 || mode == 3) {
            auto gpu_start = chrono::high_resolution_clock::now();

            // Pipeline GPU-only: frame permanece en VRAM
            d_frame.upload(frame);
            cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);
            cuda::resize(d_gray, d_resized, Size(320, 240));

            vector<Rect> hog_detections;
            vector<double> weights;
            
            hog_gpu->detectMultiScale(d_resized, hog_detections, &weights);

            // Escalar detecciones a tamaño original
            for (size_t i = 0; i < hog_detections.size(); i++) {
                Rect r = hog_detections[i];
                r.x = r.x * 2;
                r.y = r.y * 2;
                r.width = r.width * 2;
                r.height = r.height * 2;
                
                // Filtro de aspect ratio
                double aspect = (double)r.height / r.width;
                if (aspect >= 1.5 && aspect <= 3.0) {
                    all_detections.push_back(r);
                    all_labels.push_back("De pie");
                    all_confidences.push_back(1.0);
                }
            }

            auto gpu_end = chrono::high_resolution_clock::now();
            total_gpu_time += chrono::duration<double, milli>(gpu_end - gpu_start).count();
        }

        // ============ MODO 2 o 3: LBP CPU para personas agachadas ============
        if (mode == 2 || mode == 3) {
            auto cpu_start = chrono::high_resolution_clock::now();

            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            
            vector<Rect> lbp_detections;
            cascade_crouching.detectMultiScale(
                gray, 
                lbp_detections,
                1.04,            // Scale factor balanceado
                16,              // minNeighbors BALANCEADO (no tan bajo)
                CASCADE_SCALE_IMAGE,
                Size(65, 55),    // minSize razonable
                Size(400, 350)   // maxSize contenido
            );

            for (Rect r : lbp_detections) {
                // Filtro de aspect ratio BALANCEADO
                double aspect = (double)r.width / r.height;
                if (aspect < 0.5 || aspect > 2.5) continue;

                // EXPANDIR BOUNDING BOX MODERADAMENTE
                int expandX = (int)(r.width * 0.12);      // 12% más ancho
                int expandTop = (int)(r.height * 0.10);   // 10% hacia arriba
                int expandBottom = (int)(r.height * 0.25); // 25% hacia abajo (pies)
                
                r.x = max(0, r.x - expandX);
                r.y = max(0, r.y - expandTop);
                r.width = min(frame.cols - r.x, r.width + 2 * expandX);
                r.height = min(frame.rows - r.y, r.height + expandTop + expandBottom);

                Mat roi = frame(r);
                if (!isValidPerson(roi)) continue;

                all_detections.push_back(r);
                all_labels.push_back("Agachado");
                all_confidences.push_back(1.0);
            }

            auto cpu_end = chrono::high_resolution_clock::now();
            total_cpu_time += chrono::duration<double, milli>(cpu_end - cpu_start).count();
        }

        // Filtrar detecciones superpuestas
        filterOverlappingDetections(all_detections, all_labels, all_confidences, 0.35);

        // Dibujar detecciones
        for (size_t i = 0; i < all_detections.size(); i++) {
            Scalar color = (all_labels[i] == "De pie") ? Scalar(0, 255, 0) : Scalar(255, 0, 255);
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
        auto frame_end = chrono::high_resolution_clock::now();
        double frame_time = chrono::duration<double, milli>(frame_end - frame_start).count();
        double fps = 1000.0 / frame_time;
        frame_count++;

        // Mostrar información + métricas GPU
        string modeStr = (mode == 1) ? "HOG GPU" : (mode == 2) ? "LBP CPU" : "HIBRIDO";
        putText(display, "Modo: " + modeStr, Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        putText(display, "FPS: " + to_string((int)fps), Point(10, 60), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        putText(display, "Detecciones: " + to_string(all_detections.size()), 
               Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        
        if (frame_count > 0) {
            double avg_gpu = total_gpu_time / frame_count;
            double avg_cpu = total_cpu_time / frame_count;
            putText(display, "GPU: " + to_string((int)avg_gpu) + "ms", Point(10, 120), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 255), 2);
            putText(display, "CPU: " + to_string((int)avg_cpu) + "ms", Point(10, 150), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 165, 0), 2);
        }
        
        putText(display, botEnabled ? "Bot: ON" : "Bot: OFF", Point(10, 180), 
               FONT_HERSHEY_SIMPLEX, 0.7, 
               botEnabled ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 2);

        imshow("Detector Hibrido GPU", display);

        // Auto-envío
        if (AUTO_SEND_ENABLED && botEnabled && !all_detections.empty() && cooldown == 0) {
            cout << "[AUTO-GPU] Enviando (" << all_detections.size() << " personas, " 
                 << (int)fps << " FPS)" << endl;
            if (sendImageToBot(display)) {
                cooldown = COOLDOWN_FRAMES;
            }
        }

        if (cooldown > 0) cooldown--;

        // Control de teclado
        int key = waitKey(1);
        if (key == 'q' || key == 'Q') break;
        if (key == '1') { mode = 1; cout << "Modo: HOG GPU (solo de pie)" << endl; }
        if (key == '2') { mode = 2; cout << "Modo: LBP CPU (solo agachados)" << endl; }
        if (key == '3') { mode = 3; cout << "Modo: HIBRIDO GPU+CPU" << endl; }
        if (key == 'b' || key == 'B') {
            botEnabled = !botEnabled;
            cout << (botEnabled ? "Bot ACTIVADO" : "Bot DESACTIVADO") << endl;
        }
        if (key == ' ') {
            if (botEnabled && !all_detections.empty()) {
                sendImageToBot(display);
                cooldown = COOLDOWN_FRAMES;
            }
        }

        frameCounter++;
    }

    // Estadísticas finales
    cout << "\n========================================" << endl;
    cout << "  ESTADISTICAS GPU" << endl;
    cout << "========================================" << endl;
    cout << "Frames procesados: " << frame_count << endl;
    cout << "Tiempo promedio GPU: " << total_gpu_time / frame_count << " ms" << endl;
    cout << "Tiempo promedio CPU: " << total_cpu_time / frame_count << " ms" << endl;
    cout << "Speedup GPU: " << (total_cpu_time / total_gpu_time) << "x" << endl;
    cout << "========================================\n" << endl;

    cap.release();
    destroyAllWindows();
    return 0;
}
