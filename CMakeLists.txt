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

# BGS Library
set(BGSLIBRARY_INCLUDE_DIR /home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/package_bgs)
include_directories(${BGSLIBRARY_INCLUDE_DIR})

# Project
include_directories(inc/)

# Create targets
add_executable( run_stereomatcher src/Stereomatcher.cpp
    src/imageloader/BaseImageloader.cpp
    src/preprocess/BasePreprocessor.cpp
    src/bgsubtraction/CustomPixelBasedAdaptiveSegmenter.cpp
    src/bgsubtraction/CustomFrameDifference.cpp
    src/stereomatch/BasicBlockmatcher.cpp
    src/postprocess/BasePostprocessor.cpp
    src/stereomatch/BasicSGMatcher.cpp
    src/stereomatch/BasicBPMatcher.cpp
    src/ImageControl.cpp )

add_executable( run_bgs src/BackgroundSubtraction.cpp)
add_executable( run_dbscan src/dbscan.cpp )
add_executable( run_evalbackground src/EvalBackground.cpp )
add_executable( run_customblockmatcher src/CustomBlockmatcher.cpp )
add_executable( run_testbaseimageloader src/TestBaseImageloader.cpp )

target_link_libraries( run_stereomatcher ${OpenCV_LIBS} "/home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/build/libbgs.a")
target_link_libraries( run_bgs ${OpenCV_LIBS} "/home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/build/libbgs.a")
target_link_libraries( run_dbscan ${OpenCV_LIBS} )
target_link_libraries( run_evalbackground ${OpenCV_LIBS} "/home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/build/libbgs.a")
target_link_libraries( run_customblockmatcher ${OpenCV_LIBS} )
target_link_libraries( run_testbaseimageloader ${OpenCV_LIBS} )