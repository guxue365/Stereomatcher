cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
project( Stereomatcher )

# C++
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set (OpenCV_DIR /opt/opencv/opencv-3.4.1_with_vtk/share/OpenCV)
find_package(OpenCV 3.0 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} ) 

# Boost
#set(Boost_USE_STATIC_LIBS OFF) 
#set(Boost_USE_MULTITHREADED ON)  
#set(Boost_USE_STATIC_RUNTIME OFF)
#set(BOOST_ROOT /opt/boost/boost-1.67.0)
#find_package(Boost REQUIRED COMPONENTS system filesystem) 
#include_directories(${Boost_INCLUDE_DIRS}) 

# Json
set(NLOHMANN_JSON_INCLUDE_DIR /home/jung/2018EntwicklungStereoalgorithmus/nlohmann_json/include)
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# BGS Library
set(BGSLIBRARY_INCLUDE_DIR /home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/package_bgs)
include_directories(${BGSLIBRARY_INCLUDE_DIR})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} )

# PCL
find_package(PCL 1.3 REQUIRED COMPONENTS common visualization features segmentation kdtree)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# VTK
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Project
include_directories(inc/)

# VTK bugfix
list(REMOVE_ITEM PCL_VISUALIZATION_LIBRARIES "vtkproj4")
list(REMOVE_ITEM PCL_KDTREE_LIBRARIES "vtkproj4")
list(REMOVE_ITEM PCL_IO_LIBRARIES "vtkproj4")
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")


# Create targets
add_executable( run_stereomatcher src/Stereomatcher.cpp
    src/imageloader/BaseImageloader.cpp
    src/imageloader/CustomImageloader.cpp
    src/preprocess/BasePreprocessor.cpp
    src/preprocess/PreprocessMask.cpp
    src/bgsubtraction/CustomPixelBasedAdaptiveSegmenter.cpp
    src/bgsubtraction/CustomFrameDifference.cpp
    src/stereomatch/BasicBlockMatcher.cpp
    src/stereomatch/BasicSGMatcher.cpp
    src/stereomatch/BasicBPMatcher.cpp
    src/stereomatch/CustomBlockMatcher.cpp
    src/stereomatch/CustomCannyMatcher.cpp
    src/stereomatch/CustomDiffMatcher.cpp
    src/stereomatch/CustomPyramidMatcher.cpp
    src/stereomatch/BasicBPMatcher.cpp
    src/segmentation/RegionGrowing.cpp
    src/segmentation/DBSCAN.cpp
    src/segmentation/PCLSegmentation.cpp
    src/postprocess/BasePostprocessor.cpp
    src/postprocess/PostInterpolation.cpp
    src/ImageControl.cpp )

add_executable( run_bgs src/BackgroundSubtraction.cpp)
add_executable( run_dbscan src/dbscan.cpp )
add_executable( run_evalbackground src/EvalBackground.cpp )
add_executable( run_evaldisparity src/EvalDisparity.cpp
    src/stereomatch/BasicBlockMatcher.cpp
    src/stereomatch/BasicSGMatcher.cpp
    src/stereomatch/BasicBPMatcher.cpp
    src/stereomatch/CustomBlockMatcher.cpp
    src/stereomatch/CustomCannyMatcher.cpp
    src/stereomatch/CustomDiffMatcher.cpp
    src/stereomatch/CustomPyramidMatcher.cpp
    src/stereomatch/CustomMultiBoxMatcher.cpp
    src/stereomatch/CustomBlockCannyMatcher.cpp
    src/postprocess/BasePostprocessor.cpp
    src/postprocess/PostInterpolation.cpp
    src/evaluation/EvaluateBPP.cpp
    src/evaluation/EvaluateRMS.cpp )
add_executable( run_customblockmatcher src/CustomBlockmatcher.cpp )
add_executable( run_testbaseimageloader src/TestBaseImageloader.cpp )
add_executable( run_stereorectify src/Stereorectify.cpp )
add_executable( run_stereocalibrate src/Stereocalibrate.cpp )
add_executable( run_perspectivetransform src/PerspectiveTransform.cpp )
add_executable( run_testviz src/Testviz.cpp )
add_executable( run_testpcl src/TestPCL.cpp )
add_executable( run_kittiimprove src/KittiImprovement.cpp )
add_executable( run_pyramidmatcher src/PyramidBlockMatcher.cpp )
#add_executable( run_twostepsegmentation src/TwoStepSegmentation.cpp
#    src/segmentation/RegionGrowing.cpp
#    src/segmentation/DBSCAN.cpp )
add_executable (run_creategt src/CreateGT.cpp src/FileGT.cpp )

target_link_libraries( run_stereomatcher ${OpenCV_LIBS} ${PCL_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} ${PCL_KDTREE_LIBRARIES} "/home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/build/libbgs.a")
target_link_libraries( run_bgs ${OpenCV_LIBS} "/home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/build/libbgs.a")
target_link_libraries( run_dbscan ${OpenCV_LIBS} )
target_link_libraries( run_evalbackground ${OpenCV_LIBS} "/home/jung/2018EntwicklungStereoalgorithmus/bgslibrary/build/libbgs.a")
target_link_libraries( run_evaldisparity ${OpenCV_LIBS} )
target_link_libraries( run_customblockmatcher ${OpenCV_LIBS} )
target_link_libraries( run_testbaseimageloader ${OpenCV_LIBS} )
target_link_libraries( run_stereorectify ${OpenCV_LIBS} )
target_link_libraries( run_stereocalibrate ${OpenCV_LIBS} )
target_link_libraries( run_perspectivetransform ${OpenCV_LIBS} )
target_link_libraries( run_testviz ${OpenCV_LIBS} )
target_link_libraries( run_testpcl ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} )
target_link_libraries( run_kittiimprove ${OpenCV_LIBS} )
target_link_libraries( run_pyramidmatcher ${OpenCV_LIBS} )
#target_link_libraries( run_twostepsegmentation ${OpenCV_LIBS} )
target_link_libraries( run_creategt ${OpenCV_LIBS} ${PCL_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} ${PCL_KDTREE_LIBRARIES} )
