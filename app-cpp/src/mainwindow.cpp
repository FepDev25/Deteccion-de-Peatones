#include "mainwindow.h"
#include <QImage>
#include <QPixmap>
#include <QMessageBox>
#include <QCheckBox>
#include <QDebug>
#include <curl/curl.h>
#include <chrono>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    cameraRunning = false;
    botAutoSend = true;
    currentMode = 2; // Híbrido por defecto
    frameCount = 0;
    detectionCount = 0;
    cooldownFrames = 0;
    fps = 0.0;
    
    setupUI();
    initDetectors();
    
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);
}

MainWindow::~MainWindow() {
    if (camera.isOpened()) {
        camera.release();
    }
}

void MainWindow::setupUI() {
    setWindowTitle("Sistema de Detección de Peatones - Visión por Computador");
    setMinimumSize(1200, 700);

    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

    // ===== PANEL IZQUIERDO: VIDEO =====
    QVBoxLayout *videoLayout = new QVBoxLayout();
    
    videoLabel = new QLabel(this);
    videoLabel->setMinimumSize(640, 480);
    videoLabel->setStyleSheet("QLabel { background-color: black; border: 2px solid #2196F3; }");
    videoLabel->setAlignment(Qt::AlignCenter);
    videoLabel->setText("Cámara detenida\nPresiona 'Iniciar Cámara'");
    videoLabel->setStyleSheet("QLabel { color: white; font-size: 16px; background-color: #1e1e1e; border: 2px solid #2196F3; }");
    
    videoLayout->addWidget(videoLabel);
    
    // ===== PANEL DERECHO: CONTROLES =====
    QVBoxLayout *controlLayout = new QVBoxLayout();
    
    // Grupo de información
    QGroupBox *infoGroup = new QGroupBox("Información en Tiempo Real");
    infoGroup->setStyleSheet("QGroupBox { font-weight: bold; color: #2196F3; }");
    QVBoxLayout *infoLayout = new QVBoxLayout();
    
    fpsLabel = new QLabel("FPS: 0");
    fpsLabel->setStyleSheet("QLabel { font-size: 14px; padding: 5px; }");
    detectionCountLabel = new QLabel("Detecciones: 0");
    detectionCountLabel->setStyleSheet("QLabel { font-size: 14px; padding: 5px; }");
    modeLabel = new QLabel("Modo: Híbrido (HOG+LBP)");
    modeLabel->setStyleSheet("QLabel { font-size: 14px; padding: 5px; font-weight: bold; color: #4CAF50; }");
    statusLabel = new QLabel("Estado: Detenido");
    statusLabel->setStyleSheet("QLabel { font-size: 14px; padding: 5px; color: #FF5722; }");
    
    infoLayout->addWidget(fpsLabel);
    infoLayout->addWidget(detectionCountLabel);
    infoLayout->addWidget(modeLabel);
    infoLayout->addWidget(statusLabel);
    infoGroup->setLayout(infoLayout);
    
    // Grupo de selección de modo
    QGroupBox *modeGroup = new QGroupBox("Pipeline de Detección");
    modeGroup->setStyleSheet("QGroupBox { font-weight: bold; color: #2196F3; }");
    QVBoxLayout *modeLayout = new QVBoxLayout();
    
    QLabel *modeSelectLabel = new QLabel("Seleccionar técnica:");
    modeComboBox = new QComboBox();
    modeComboBox->addItem("🟢 HOG + SVM (Solo de pie)");
    modeComboBox->addItem("🟣 LBP Cascades (Agachados)");
    modeComboBox->addItem("🔵 Híbrido (HOG + LBP)");
    modeComboBox->setCurrentIndex(2);
    modeComboBox->setStyleSheet("QComboBox { padding: 8px; font-size: 13px; }");
    connect(modeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &MainWindow::changeDetectionMode);
    
    modeLayout->addWidget(modeSelectLabel);
    modeLayout->addWidget(modeComboBox);
    modeGroup->setLayout(modeLayout);
    
    // Grupo de controles principales
    QGroupBox *controlGroup = new QGroupBox("Controles");
    controlGroup->setStyleSheet("QGroupBox { font-weight: bold; color: #2196F3; }");
    QVBoxLayout *buttonLayout = new QVBoxLayout();
    
    startStopButton = new QPushButton("▶️ Iniciar Cámara");
    startStopButton->setStyleSheet(
        "QPushButton { background-color: #4CAF50; color: white; font-size: 16px; "
        "padding: 12px; border-radius: 5px; font-weight: bold; }"
        "QPushButton:hover { background-color: #45a049; }"
    );
    connect(startStopButton, &QPushButton::clicked, this, &MainWindow::toggleCamera);
    
    sendBotButton = new QPushButton("📤 Enviar a Telegram");
    sendBotButton->setStyleSheet(
        "QPushButton { background-color: #2196F3; color: white; font-size: 14px; "
        "padding: 10px; border-radius: 5px; }"
        "QPushButton:hover { background-color: #0b7dda; }"
        "QPushButton:disabled { background-color: #cccccc; }"
    );
    sendBotButton->setEnabled(false);
    connect(sendBotButton, &QPushButton::clicked, this, &MainWindow::sendToBot);
    
    autoSendCheckBox = new QCheckBox("Auto-envío al bot (cada 2 seg)");
    autoSendCheckBox->setChecked(true);
    autoSendCheckBox->setStyleSheet("QCheckBox { font-size: 13px; padding: 5px; }");
    connect(autoSendCheckBox, &QCheckBox::toggled, this, &MainWindow::toggleBotAutoSend);
    
    buttonLayout->addWidget(startStopButton);
    buttonLayout->addWidget(sendBotButton);
    buttonLayout->addWidget(autoSendCheckBox);
    controlGroup->setLayout(buttonLayout);
    
    // Grupo de información adicional
    QGroupBox *helpGroup = new QGroupBox("Información del Sistema");
    helpGroup->setStyleSheet("QGroupBox { font-weight: bold; color: #2196F3; }");
    QVBoxLayout *helpLayout = new QVBoxLayout();
    
    QLabel *helpText = new QLabel(
        "<b>HOG + SVM:</b> Alta precisión para personas de pie<br>"
        "<b>LBP Cascades:</b> Detecta agachados/sentados<br>"
        "<b>Híbrido:</b> Combina ambas técnicas<br><br>"
        "<span style='color: #4CAF50;'>Verde</span> = De pie | "
        "<span style='color: #9C27B0;'>Morado</span> = Agachado"
    );
    helpText->setWordWrap(true);
    helpText->setStyleSheet("QLabel { font-size: 12px; padding: 10px; }");
    
    helpLayout->addWidget(helpText);
    helpGroup->setLayout(helpLayout);
    
    // Agregar todos los grupos al layout de control
    controlLayout->addWidget(infoGroup);
    controlLayout->addWidget(modeGroup);
    controlLayout->addWidget(controlGroup);
    controlLayout->addWidget(helpGroup);
    controlLayout->addStretch();
    
    // Agregar layouts principales
    mainLayout->addLayout(videoLayout, 2);
    mainLayout->addLayout(controlLayout, 1);
}

void MainWindow::initDetectors() {
    // Inicializar HOG con detector pre-entrenado
    hogDetector.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    
    // Cargar cascadas LBP
    if (!cascadeStanding.load("cascade_standing.xml")) {
        QMessageBox::warning(this, "Advertencia", 
            "No se pudo cargar cascade_standing.xml\nLBP no estará disponible");
    }
    
    if (!cascadeCrouching.load("cascade_crouching.xml")) {
        QMessageBox::warning(this, "Advertencia", 
            "No se pudo cargar cascade_crouching.xml\nDetección de agachados no disponible");
    }
}

void MainWindow::toggleCamera() {
    if (!cameraRunning) {
        camera.open(0);
        if (!camera.isOpened()) {
            QMessageBox::critical(this, "Error", "No se pudo abrir la cámara");
            return;
        }
        
        camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        
        cameraRunning = true;
        startStopButton->setText("⏸️ Detener Cámara");
        startStopButton->setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-size: 16px; "
            "padding: 12px; border-radius: 5px; font-weight: bold; }"
            "QPushButton:hover { background-color: #E64A19; }"
        );
        sendBotButton->setEnabled(true);
        statusLabel->setText("Estado: ✅ Activo");
        statusLabel->setStyleSheet("QLabel { font-size: 14px; padding: 5px; color: #4CAF50; }");
        
        fpsStartTime = std::chrono::steady_clock::now();
        frameCount = 0;
        
        timer->start(30); // ~33 FPS
    } else {
        timer->stop();
        camera.release();
        cameraRunning = false;
        startStopButton->setText("▶️ Iniciar Cámara");
        startStopButton->setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 16px; "
            "padding: 12px; border-radius: 5px; font-weight: bold; }"
            "QPushButton:hover { background-color: #45a049; }"
        );
        sendBotButton->setEnabled(false);
        statusLabel->setText("Estado: ⏸️ Detenido");
        statusLabel->setStyleSheet("QLabel { font-size: 14px; padding: 5px; color: #FF5722; }");
        
        videoLabel->clear();
        videoLabel->setText("Cámara detenida\nPresiona 'Iniciar Cámara'");
        videoLabel->setStyleSheet("QLabel { color: white; font-size: 16px; background-color: #1e1e1e; border: 2px solid #2196F3; }");
    }
}

void MainWindow::updateFrame() {
    if (!cameraRunning) return;
    
    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) return;
    
    processFrame();
}

void MainWindow::processFrame() {
    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) return;
    
    cv::Mat display = frame.clone();
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Rect> detections;
    std::vector<std::string> labels;
    std::vector<double> confidences;
    
    // Ejecutar detección según modo seleccionado
    switch(currentMode) {
        case 0: // HOG solo
            detectHOG(frame, gray, detections, labels, confidences);
            break;
        case 1: // LBP solo
            detectLBP(frame, gray, detections, labels, confidences);
            break;
        case 2: // Híbrido
            detectHybrid(frame, gray, detections, labels, confidences);
            break;
    }
    
    // Filtrar superposiciones
    filterOverlapping(detections, labels, confidences);
    
    // Dibujar detecciones
    for (size_t i = 0; i < detections.size(); i++) {
        cv::Scalar color = (labels[i] == "Peaton detectado") ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 255);
        cv::rectangle(display, detections[i], color, 2);
        
        int baseline;
        cv::Size textSize = cv::getTextSize(labels[i], cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Point textOrg(detections[i].x, detections[i].y - 10);
        
        cv::rectangle(display, textOrg + cv::Point(0, baseline), 
                     textOrg + cv::Point(textSize.width, -textSize.height), 
                     color, cv::FILLED);
        cv::putText(display, labels[i], textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                   cv::Scalar(0, 0, 0), 2);
    }
    
    // Actualizar estadísticas
    frameCount++;
    detectionCount = detections.size();
    
    auto currentTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(currentTime - fpsStartTime).count();
    if (elapsed >= 1.0) {
        fps = frameCount / elapsed;
        frameCount = 0;
        fpsStartTime = currentTime;
    }
    
    // Actualizar UI
    fpsLabel->setText(QString("FPS: %1").arg(static_cast<int>(fps)));
    detectionCountLabel->setText(QString("Detecciones: %1").arg(detectionCount));
    
    // Auto-envío al bot (igual que main_hybrid.cpp - cada 60 frames)
    if (botAutoSend && detectionCount > 0 && cooldownFrames == 0) {
        qDebug() << "[AUTO] Enviando detección al bot...";
        if (sendImageViaHTTP(display)) {
            qDebug() << "[AUTO] Imagen enviada exitosamente!" << detectionCount << "personas";
            cooldownFrames = 60; // 2 segundos a ~30 FPS
        }
    }
    if (cooldownFrames > 0) cooldownFrames--;
    
    // Convertir a QImage y mostrar
    cv::cvtColor(display, display, cv::COLOR_BGR2RGB);
    QImage qimg(display.data, display.cols, display.rows, display.step, QImage::Format_RGB888);
    videoLabel->setPixmap(QPixmap::fromImage(qimg).scaled(videoLabel->size(), Qt::KeepAspectRatio));
}

void MainWindow::detectHOG(cv::Mat& frame, cv::Mat& gray, std::vector<cv::Rect>& detections, 
                           std::vector<std::string>& labels, std::vector<double>& confidences) {
    std::vector<cv::Rect> hogDets;
    std::vector<double> weights;
    
    // IMPORTANTE: Reducir tamaño para mejor performance (igual que main_hybrid.cpp)
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(320, 240));
    
    // Parámetros EXACTOS de main_hybrid.cpp (funciona perfectamente)
    hogDetector.detectMultiScale(resized, hogDets, weights, 
                                 0.0,           // hitThreshold (balanceado)
                                 cv::Size(4,4), // winStride
                                 cv::Size(8,8), // padding
                                 1.05,          // scale
                                 2.0);          // finalThreshold
    
    for (size_t i = 0; i < hogDets.size(); i++) {
        cv::Rect r = hogDets[i];
        r.x *= 2;
        r.y *= 2;
        r.width *= 2;
        r.height *= 2;
        
        // Filtro de aspect ratio (personas de pie)
        double aspect = (double)r.height / r.width;
        if (aspect >= 1.5 && aspect <= 3.0) {
            detections.push_back(r);
            labels.push_back("Peaton detectado");
            confidences.push_back(weights[i]);
        }
    }
}

void MainWindow::detectLBP(cv::Mat& frame, cv::Mat& gray, std::vector<cv::Rect>& detections, 
                           std::vector<std::string>& labels, std::vector<double>& confidences) {
    std::vector<cv::Rect> lbpDets;
    
    // Parámetros BALANCEADOS: detectar agachados sin falsos positivos
    cascadeCrouching.detectMultiScale(
        gray, 
        lbpDets,
        1.05,              // scaleFactor (más escalas)
        18,                // minNeighbors BALANCEADO
        cv::CASCADE_SCALE_IMAGE,
        cv::Size(75, 65),  // minSize (personas agachadas pequeñas)
        cv::Size(320, 300) // maxSize (personas sentadas grandes)
    );
    
    for (cv::Rect r : lbpDets) {
        // Filtro de aspect ratio (agachados/sentados/parciales)
        double aspect = (double)r.width / r.height;
        if (aspect < 0.6 || aspect > 2.3) continue;
        
        // FILTRAR SI YA HAY DETECCIÓN HOG CERCANA (evitar duplicados)
        bool overlapWithHOG = false;
        for (size_t i = 0; i < detections.size(); i++) {
            if (labels[i] == "Peaton detectado") {
                cv::Rect intersection = r & detections[i];
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
        
        r.x = std::max(0, r.x - expandX);
        r.y = std::max(0, r.y - expandTop);
        r.width = std::min(frame.cols - r.x, r.width + 2 * expandX);
        r.height = std::min(frame.rows - r.y, r.height + expandTop + expandBottom);
        
        // Validación inteligente de ROI (filtra ventanas automáticamente)
        cv::Mat roi = frame(r);
        if (!isValidPerson(roi)) continue;
        
        detections.push_back(r);
        labels.push_back("Peaton detectado");
        confidences.push_back(1.0); // LBP no da scores
    }
}

void MainWindow::detectHybrid(cv::Mat& frame, cv::Mat& gray, std::vector<cv::Rect>& detections, 
                              std::vector<std::string>& labels, std::vector<double>& confidences) {
    detectHOG(frame, gray, detections, labels, confidences);
    detectLBP(frame, gray, detections, labels, confidences);
}

void MainWindow::filterOverlapping(std::vector<cv::Rect>& detections, std::vector<std::string>& labels, 
                                   std::vector<double>& confidences) {
    if (detections.empty()) return;
    
    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < detections.size(); i++) indices[i] = i;
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return confidences[a] > confidences[b];
    });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<cv::Rect> filtered;
    std::vector<std::string> filteredLabels;
    std::vector<double> filteredConf;
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        filtered.push_back(detections[idx]);
        filteredLabels.push_back(labels[idx]);
        filteredConf.push_back(confidences[idx]);
        
        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            cv::Rect inter = detections[idx] & detections[idx2];
            double iou = inter.area() / (double)(detections[idx].area() + detections[idx2].area() - inter.area());
            
            if (iou > 0.4) suppressed[idx2] = true;
        }
    }
    
    detections = filtered;
    labels = filteredLabels;
    confidences = filteredConf;
}

bool MainWindow::isValidPerson(const cv::Mat& roi) {
    // Validación inteligente para LBP (eliminar ventanas sin perder personas)
    cv::Mat gray_roi;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
    } else {
        gray_roi = roi.clone();
    }
    
    // 1. Varianza de color - ventanas son muy uniformes, personas tienen textura
    cv::Scalar mean_color, stddev_color;
    cv::meanStdDev(gray_roi, mean_color, stddev_color);
    if (stddev_color[0] < 17.0) {  // Aumentado para filtrar sillas uniformes
        return false; // Demasiado uniforme (ventana/pared/silla)
    }
    
    // 2. Canny edge detection - densidad de bordes
    cv::Mat edges;
    cv::Canny(gray_roi, edges, 50, 150);
    double edgeDensity = cv::countNonZero(edges) / (double)(edges.rows * edges.cols);
    
    // Ventanas/sillas tienen baja densidad O demasiada (marcos metálicos)
    if (edgeDensity < 0.08 || edgeDensity > 0.40) {  // Reducido máximo para filtrar sillas
        return false;
    }
    
    // 3. Sobel gradients - verificar estructura humana
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray_roi, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(gray_roi, grad_y, CV_16S, 0, 1, 3);
    
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    cv::Scalar mean_x, stddev_x, mean_y, stddev_y;
    cv::meanStdDev(abs_grad_x, mean_x, stddev_x);
    cv::meanStdDev(abs_grad_y, mean_y, stddev_y);
    
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
            diff += std::abs(center - gray_roi.at<uchar>(i-1, j)) > 10 ? 1 : 0;
            diff += std::abs(center - gray_roi.at<uchar>(i+1, j)) > 10 ? 1 : 0;
            diff += std::abs(center - gray_roi.at<uchar>(i, j-1)) > 10 ? 1 : 0;
            diff += std::abs(center - gray_roi.at<uchar>(i, j+1)) > 10 ? 1 : 0;
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
    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
    cv::Mat skin_mask;
    cv::inRange(hsv_roi, cv::Scalar(0, 20, 60), cv::Scalar(20, 255, 255), skin_mask);
    double skinRatio = cv::countNonZero(skin_mask) / (double)(skin_mask.rows * skin_mask.cols);
    if (skinRatio < 0.012) {  // Al menos 1.2% de píxeles con tono piel
        return false;
    }
    
    return true;
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

bool MainWindow::sendImageViaHTTP(const cv::Mat& frame) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    
    std::vector<uchar> buf;
    cv::imencode(".jpg", frame, buf);
    
    struct curl_httppost* formpost = NULL;
    struct curl_httppost* lastptr = NULL;
    
    curl_formadd(&formpost, &lastptr,
                 CURLFORM_COPYNAME, "image",
                 CURLFORM_BUFFER, "detection.jpg",
                 CURLFORM_BUFFERPTR, buf.data(),
                 CURLFORM_BUFFERLENGTH, buf.size(),
                 CURLFORM_CONTENTTYPE, "image/jpeg",
                 CURLFORM_END);
    
    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:5000/detect");
    curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    CURLcode res = curl_easy_perform(curl);
    bool success = (res == CURLE_OK);
    
    curl_easy_cleanup(curl);
    curl_formfree(formpost);
    
    return success;
}

void MainWindow::sendToBot() {
    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) return;
    
    if (sendImageViaHTTP(frame)) {
        QMessageBox::information(this, "Éxito", "Imagen enviada a Telegram");
    } else {
        QMessageBox::warning(this, "Error", "No se pudo enviar la imagen");
    }
}

void MainWindow::changeDetectionMode(int index) {
    currentMode = index;
    QString modeText;
    switch(index) {
        case 0:
            modeText = "Modo: HOG + SVM (Solo de pie)";
            break;
        case 1:
            modeText = "Modo: LBP Cascades (Agachados)";
            break;
        case 2:
            modeText = "Modo: Híbrido (HOG + LBP)";
            break;
    }
    modeLabel->setText(modeText);
}

void MainWindow::toggleBotAutoSend(bool checked) {
    botAutoSend = checked;
}
