cmake_minimum_required(VERSION 2.8.0)

project(wksuper-proposals)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.2.0 REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(edge_box/cpp)

set(EDGEBOX_CPP "${PROJECT_SOURCE_DIR}/edge_box/cpp")
set(EDGEBOX_GOP_BUILD_LIB "${EDGEBOX_CPP}/external/gop_1.3/build/lib")
set(EDGEBOX_GOP_LIB "${EDGEBOX_CPP}/external/gop_1.3/lib")
message(${EDGEBOX_GOP_BUILD_LIB})
include_directories(${OpenCV_INCLUDE_DIRS} ${EDGEBOX_CPP}/include ${EDGEBOX_GOP_LIB} ${EIGEN3_INCLUDE_DIR})
link_directories(${OpenCV_LIB_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # gop need c++11
add_executable(edge_box_main edge_box_main.cpp edge_box/cpp/src/edge_boxes.cpp edge_box/cpp/src/edge_detect.cpp ${EDGEBOX_CPP}/src/edge_nms.cpp ${EDGEBOX_CPP}/src/util.cpp)
target_link_libraries(edge_box_main ${OpenCV_LIBS} ${EDGEBOX_GOP_BUILD_LIB}/contour/libcontour.a ${EDGEBOX_GOP_BUILD_LIB}/imgproc/libimgproc.a ${EDGEBOX_GOP_BUILD_LIB}/learning/liblearning.a ${EDGEBOX_GOP_BUILD_LIB}/util/libutil.a ${EDGEBOX_GOP_BUILD_LIB}/proposals/libproposals.a)
