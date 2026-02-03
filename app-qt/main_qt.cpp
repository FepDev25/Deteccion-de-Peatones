#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    // Estilo moderno para la aplicación
    app.setStyle("Fusion");
    
    MainWindow window;
    window.show();
    
    return app.exec();
}
