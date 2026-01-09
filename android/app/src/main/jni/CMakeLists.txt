project(YOLOv5ss)

cmake_minimum_required(VERSION 3.10.2)

set(OpenCV_DIR D:\\AndroidStudioProjects\\ncnn\\opencv-mobile-4.5.1-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR D:\\AndroidStudioProjects\\ncnn/ncnn-20220420-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(ncnnv5lite SHARED yolov5ncnn.cpp yolov5.cpp ndkcamera.cpp)

target_link_libraries(ncnnv5lite ncnn ${OpenCV_LIBS} camera2ndk mediandk)
