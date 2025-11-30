cmake_minimum_required(VERSION 3.19)
project(CNN_HWDR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets Concurrent)
find_package(Qt6 REQUIRED COMPONENTS Core)

find_package(OpenMP REQUIRED)

set(EIGEN3_INCLUDE_DIR "D:/Program/eigen-3.4.0")
add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
)

find_package(OpenCV REQUIRED CONFIGS PATHS "D:/Program/opencv-4.5.5-mingw/x64/mingw/lib")
include_directories(${OpenCV_INCLUDE_DIRS})

qt_standard_project_setup()

qt_add_executable(CNN_HWDR
    WIN32 MACOSX_BUNDLE
    main.cpp
    graphicscore.cpp graphicscore.h graphicscore.ui
    image_process.cpp image_process.hpp
    mnist_data.cpp mnist_data.hpp
    activation_function.cpp activation_function.hpp
    filter.cpp filter.hpp pool_layer.cpp pool_layer.hpp
    convolve_layer.cpp convolve_layer.hpp
    full_connect_layer.cpp full_connect_layer.hpp
    le_net5.cpp le_net5.hpp vgg16.cpp vgg16.hpp
    cnn_base.h cnn_base.cpp
    language_manager.h language_manager.cpp

)

target_link_libraries(CNN_HWDR
    PRIVATE
        Qt::Core
        Qt::Widgets
        Qt::Concurrent
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        ${OpenCV_LIBS}
)
target_link_libraries(CNN_HWDR PRIVATE Qt6::Core)

include(GNUInstallDirs)

install(TARGETS CNN_HWDR
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET CNN_HWDR
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
