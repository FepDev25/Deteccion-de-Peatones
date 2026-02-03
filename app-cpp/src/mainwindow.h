#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void toggleCamera();
    void updateFrame();
    void sendToBot();
    void changeDetectionMode(int index);
    void toggleBotAutoSend(bool checked);

private:
    void setupUI();
    void initDetectors();
    void processFrame();
    void detectHOG(cv::Mat& frame, cv::Mat& gray, std::vector<cv::Rect>& detections, 
                   std::vector<std::string>& labels, std::vector<double>& confidences);
    void detectLBP(cv::Mat& frame, cv::Mat& gray, std::vector<cv::Rect>& detections, 
                   std::vector<std::string>& labels, std::vector<double>& confidences);
    void detectHybrid(cv::Mat& frame, cv::Mat& gray, std::vector<cv::Rect>& detections, 
                      std::vector<std::string>& labels, std::vector<double>& confidences);
    void filterOverlapping(std::vector<cv::Rect>& detections, std::vector<std::string>& labels, 
                          std::vector<double>& confidences);
    bool isValidPerson(const cv::Mat& roi);
    bool sendImageViaHTTP(const cv::Mat& frame);

    // UI Components
    QLabel *videoLabel;
    QLabel *fpsLabel;
    QLabel *detectionCountLabel;
    QLabel *modeLabel;
    QLabel *statusLabel;
    QPushButton *startStopButton;
    QPushButton *sendBotButton;
    QComboBox *modeComboBox;
    QCheckBox *autoSendCheckBox;

    // OpenCV
    cv::VideoCapture camera;
    cv::HOGDescriptor hogDetector;
    cv::CascadeClassifier cascadeStanding;
    cv::CascadeClassifier cascadeCrouching;
    
    // Timer
    QTimer *timer;
    
    // State
    bool cameraRunning;
    bool botAutoSend;
    int currentMode; // 0=HOG, 1=LBP, 2=Hybrid
    int frameCount;
    int detectionCount;
    int cooldownFrames;
    double fps;
    std::chrono::steady_clock::time_point fpsStartTime;
};

#endif // MAINWINDOW_H
