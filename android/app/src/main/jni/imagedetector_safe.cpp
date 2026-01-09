#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yolov5.h"
#include <platform.h>
#include <benchmark.h>

static Yolov5* g_imageDetector = 0;
static ncnn::Mutex imageLock;

extern "C" {

// ImageDetector implementation

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_ImageDetector_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "loadModel called with modelid=%d, cpugpu=%d", (int)modelid, (int)cpugpu);
    
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Invalid parameters: modelid=%d, cpugpu=%d", (int)modelid, (int)cpugpu);
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to get AssetManager");
        return JNI_FALSE;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "AssetManager obtained: %p", mgr);

    const char* modeltypes[] = {
            "320-lite-e",
            "416-lite-e",
            "320-lite-i8e",
            "416-lite-i8e",
            "416-lite-s",
            "416-lite-i8s",
            "512-lite-c",
    };

    const int target_sizes[] = {
            320,
            416,
            320,
            416,
            416,
            416,
            512
    };

    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Loading model: %s, target_size: %d, use_gpu: %d", modeltype, target_size, use_gpu);

    // reload
    {
        ncnn::MutexLockGuard g(imageLock);
        
        // Clean up existing detector
        if (g_imageDetector) {
            delete g_imageDetector;
            g_imageDetector = 0;
        }
        
        if (use_gpu && ncnn::get_gpu_count() == 0) {
            __android_log_print(ANDROID_LOG_WARN, "ImageDetector", "GPU requested but no GPU available, using CPU");
            use_gpu = false;
        }

        g_imageDetector = new Yolov5;
        if (!g_imageDetector) {
            __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to create Yolov5 instance");
            return JNI_FALSE;
        }

        int loadResult = g_imageDetector->load(mgr, modeltype, target_size, use_gpu);
        if (loadResult != 0) {
            __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to load model: %s, error code: %d", modeltype, loadResult);
            delete g_imageDetector;
            g_imageDetector = 0;
            return JNI_FALSE;
        }
        
        __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Model loaded successfully: %s", modeltype);
    }

    return JNI_TRUE;
}

// public native Bitmap detectImage(Bitmap bitmap);
JNIEXPORT jobject JNICALL Java_ncnn_v5lite_demo_ImageDetector_detectImage(JNIEnv* env, jobject thiz, jobject bitmap)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "detectImage called");
    
    {
        ncnn::MutexLockGuard g(imageLock);
        if (g_imageDetector == 0) {
            __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Model not loaded");
            return NULL;
        }
    }
    
    if (!bitmap) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Input bitmap is null");
        return NULL;
    }
    
    // Get bitmap info
    AndroidBitmapInfo info;
    int ret = AndroidBitmap_getInfo(env, bitmap, &info);
    if (ret < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "AndroidBitmap_getInfo failed, error: %d", ret);
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Bitmap info: %dx%d, format: %d", info.width, info.height, info.format);
    
    // Check bitmap format
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Bitmap format is not RGBA_8888, format: %d", info.format);
        return NULL;
    }
    
    // Lock bitmap pixels
    void* pixels;
    ret = AndroidBitmap_lockPixels(env, bitmap, &pixels);
    if (ret < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "AndroidBitmap_lockPixels failed, error: %d", ret);
        return NULL;
    }
    
    if (!pixels) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Bitmap pixels is null");
        AndroidBitmap_unlockPixels(env, bitmap);
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Bitmap pixels locked successfully");
    
    // Create OpenCV Mat from bitmap
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    
    // Color conversion with error checking
    if (rgba.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "RGBA Mat is empty");
        AndroidBitmap_unlockPixels(env, bitmap);
        return NULL;
    }
    
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    
    if (bgr.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Color conversion failed");
        AndroidBitmap_unlockPixels(env, bitmap);
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Color conversion completed");
    
    // Unlock bitmap pixels early
    AndroidBitmap_unlockPixels(env, bitmap);
    
    // Perform detection
    std::vector<Object> objects;
    double t = 0;
    
    {
        ncnn::MutexLockGuard g(imageLock);
        if (g_imageDetector) {
            __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Starting detection...");
            t = g_imageDetector->detect(bgr, objects);
            __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Detection completed, found %d objects, time: %.2f ms", (int)objects.size(), t);
            g_imageDetector->draw(bgr, objects);
            __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Drawing completed");
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Model not loaded during detection");
            return NULL;
        }
    }
    
    // Draw detection time
    char text[64];
    sprintf(text, "Time: %.2f ms, Objects: %d", t, (int)objects.size());
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(bgr, cv::Rect(cv::Point(10, 10), 
                cv::Size(label_size.width, label_size.height + baseLine)),
                cv::Scalar(255, 255, 255), -1);
    cv::putText(bgr, text, cv::Point(10, 10 + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    // Convert back to RGBA format for Android bitmap
    cv::Mat result_rgba;
    cv::cvtColor(bgr, result_rgba, cv::COLOR_BGR2RGBA);
    
    if (result_rgba.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Result color conversion failed");
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Result color conversion completed");
    
    // Create a new bitmap to return using a simpler approach
    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    if (!bitmapClass) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to find Bitmap class");
        return NULL;
    }
    
    jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass, "createBitmap", "([IIILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    if (!createBitmapMethodID) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to find createBitmap method");
        return NULL;
    }
    
    // Get ARGB_8888 config
    jclass configClass = env->FindClass("android/graphics/Bitmap$Config");
    if (!configClass) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to find Bitmap$Config class");
        return NULL;
    }
    
    jfieldID argb8888Field = env->GetStaticFieldID(configClass, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    if (!argb8888Field) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to find ARGB_8888 field");
        return NULL;
    }
    
    jobject config = env->GetStaticObjectField(configClass, argb8888Field);
    if (!config) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to get ARGB_8888 config");
        return NULL;
    }
    
    // Convert RGBA data to int array
    int width = result_rgba.cols;
    int height = result_rgba.rows;
    int* pixels_array = new int[width * height];
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec4b pixel = result_rgba.at<cv::Vec4b>(y, x);
            // Convert RGBA to ARGB
            int argb = (pixel[3] << 24) | (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];
            pixels_array[y * width + x] = argb;
        }
    }
    
    // Create int array
    jintArray pixelArray = env->NewIntArray(width * height);
    if (!pixelArray) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to create pixel array");
        delete[] pixels_array;
        return NULL;
    }
    
    env->SetIntArrayRegion(pixelArray, 0, width * height, pixels_array);
    delete[] pixels_array;
    
    // Create bitmap
    jobject resultBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, 
                                                      pixelArray, width, height, config);
    
    // Clean up local references
    env->DeleteLocalRef(pixelArray);
    env->DeleteLocalRef(configClass);
    env->DeleteLocalRef(bitmapClass);
    
    if (!resultBitmap) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to create result bitmap");
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Detection complete, returning bitmap");
    return resultBitmap;
}

// public native void releaseModel();
JNIEXPORT void JNICALL Java_ncnn_v5lite_demo_ImageDetector_releaseModel(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "releaseModel called");
    
    ncnn::MutexLockGuard g(imageLock);
    if (g_imageDetector) {
        delete g_imageDetector;
        g_imageDetector = 0;
        __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Model released");
    }
}

} // extern "C"
