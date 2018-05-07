cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
project( Stereomatcher )

# C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set (OpenCV_DIR /opt/opencv/opencv-3.4.1/share/OpenCV)
find_package(OpenCV 3.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ROOT /opt/boost/boost-1.67.0)
find_package(Boost REQUIRED COMPONENTS system filesystem) 
include_directories(${Boost_INCLUDE_DIRS}) 

# Json
set(NLOHMANN_JSON_INCLUDE_DIR /home/jung/2018EntwicklungStereoalgorithmus/nlohmann_json/include)
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# Project
include_directories(inc/)

# Create targets
add_executable( run_stereomatcher src/main.cpp
    src/ImageHandler.cpp
    src/BasePreprocessor.cpp
    src/BasicBlockmatcher.cpp
    src/BasePostprocessor.cpp
    src/BasicSGMatcher.cpp
    src/ImageControl.cpp
    src/EvaluateRMS.cpp
    src/EvaluateBPP.cpp )

add_executable( run_testkitty src/TestStereoKitty.cpp src/EvaluateBPP.cpp src/EvaluateRMS.cpp)
add_executable( run_bgs src/BackgroundSubtraction.cpp)

target_link_libraries( run_stereomatcher ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries( run_testkitty ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries( run_bgs ${OpenCV_LIBS} ${Boost_LIBRARIES})