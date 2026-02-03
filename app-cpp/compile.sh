#!/bin/bash
# Script de compilación y ejecución rápida
# Uso: ./compile.sh [clean|run|both]

APP_DIR="/home/samidev/Escritorio/VISION-POR-COMPUTADOR/Deteccion-de-Peatones/app-cpp"

cd "$APP_DIR" || exit 1

case "$1" in
    clean)
        echo "🧹 Limpiando build anterior..."
        rm -rf build
        echo "✓ Limpieza completa"
        ;;
    
    run)
        if [ ! -f "build/app_vigilante" ]; then
            echo "❌ No existe el ejecutable. Primero compila con: ./compile.sh both"
            exit 1
        fi
        echo "🚀 Ejecutando aplicación..."
        cd build
        ./app_vigilante "$2"
        ;;
    
    both|"")
        echo "🔨 Compilando proyecto..."
        
        # Limpiar si existe
        if [ -d "build" ]; then
            echo "  Limpiando build anterior..."
            rm -rf build
        fi
        
        # Crear directorio build
        mkdir build
        cd build
        
        # Verificar dependencias
        echo "  Verificando OpenCV..."
        if ! pkg-config --exists opencv4; then
            echo "  ⚠️  OpenCV no encontrado. Instalando..."
            sudo apt install -y libopencv-dev
        fi
        
        echo "  Verificando CURL..."
        if ! pkg-config --exists libcurl; then
            echo "  ⚠️  CURL no encontrado. Instalando..."
            sudo apt install -y libcurl4-openssl-dev
        fi
        
        # Compilar
        echo "  Ejecutando CMake..."
        cmake .. || { echo "❌ Error en CMake"; exit 1; }
        
        echo "  Compilando..."
        make || { echo "❌ Error en Make"; exit 1; }
        
        echo ""
        echo "✅ ¡Compilación exitosa!"
        echo ""
        echo "📋 Para ejecutar:"
        echo "   cd build && ./app_vigilante"
        echo ""
        echo "   O usa modos:"
        echo "   ./app_vigilante standing   - Solo personas paradas"
        echo "   ./app_vigilante crouching  - Solo personas agachadas"
        echo ""
        
        # Verificar que los modelos estén copiados
        if [ -f "cascade_standing.xml" ] && [ -f "cascade_crouching.xml" ]; then
            echo "✓ Modelos .xml copiados correctamente"
        else
            echo "⚠️  Modelos no encontrados. Copiando manualmente..."
            cp ../training/models/standing/cascade_standing.xml . 2>/dev/null || echo "  ⚠️ cascade_standing.xml no encontrado"
            cp ../training/models/complex/cascade_crouching.xml . 2>/dev/null || echo "  ⚠️ cascade_crouching.xml no encontrado"
        fi
        
        echo ""
        read -p "¿Ejecutar ahora? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[YySs]$ ]]; then
            echo "🚀 Iniciando aplicación..."
            ./app_vigilante
        fi
        ;;
    
    *)
        echo "Uso: $0 [clean|run|both]"
        echo ""
        echo "Opciones:"
        echo "  clean  - Limpiar build anterior"
        echo "  run    - Ejecutar sin compilar"
        echo "  both   - Limpiar, compilar y ejecutar (por defecto)"
        echo ""
        echo "Ejemplos:"
        echo "  $0              # Compilar y ejecutar"
        echo "  $0 clean        # Solo limpiar"
        echo "  $0 run          # Solo ejecutar"
        echo "  $0 run standing # Ejecutar en modo 'standing'"
        exit 1
        ;;
esac
