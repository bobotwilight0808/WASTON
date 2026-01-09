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
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "ImageDetector loadModel %p", mgr);

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

    // reload
    {
        ncnn::MutexLockGuard g(imageLock);
        
        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // no gpu
            delete g_imageDetector;
            g_imageDetector = 0;
        } else {
            if (!g_imageDetector)
                g_imageDetector = new Yolov5;
            bool loadResult = g_imageDetector->load(mgr, modeltype, target_size, use_gpu);
            if (!loadResult) {
                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to load model: %s", modeltype);
                delete g_imageDetector;
                g_imageDetector = 0;
                return JNI_FALSE;
            }
        }
    }

    return JNI_TRUE;
}

// public native Bitmap detectImage(Bitmap bitmap);
JNIEXPORT jobject JNICALL Java_ncnn_v5lite_demo_ImageDetector_detectImage(JNIEnv* env, jobject thiz, jobject bitmap)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "detectImage called");
    
    if (g_imageDetector == 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Model not loaded");
        return NULL;
    }
    
    // Get bitmap info
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "AndroidBitmap_getInfo failed");
        return NULL;
    }
    
    // Check bitmap format
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Bitmap format is not RGBA_8888");
        return NULL;
    }
    
    // Lock bitmap pixels
    void* pixels;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "AndroidBitmap_lockPixels failed");
        return NULL;
    }
    
    // Create OpenCV Mat from bitmap
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr(info.height, info.width, CV_8UC3);
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    
    // Unlock bitmap pixels
    AndroidBitmap_unlockPixels(env, bitmap);
    
    // Perform detection
    std::vector<Object> objects;
    double t = 0;
    
    {
        ncnn::MutexLockGuard g(imageLock);
        if (g_imageDetector) {
            t = g_imageDetector->detect(bgr, objects);
            g_imageDetector->draw(bgr, objects);
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Model not loaded");
            return NULL;
        }
    }
    
    // Draw detection time
    char text[32];
    sprintf(text, "Detect time: %.2f ms", t);
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
    
    // Create a new bitmap to return
    jobject resultBitmap = NULL;
    jclass bitmapConfig = env->FindClass("android/graphics/Bitmap$Config");
    if (!bitmapConfig) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find Bitmap$Config class");
        return NULL;
    }
    
    jfieldID rgba8888FieldID = env->GetStaticFieldID(bitmapConfig, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    if (!rgba8888FieldID) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find ARGB_8888 field");
        return NULL;
    }
    
    jobject rgba8888Obj = env->GetStaticObjectField(bitmapConfig, rgba8888FieldID);
    if (!rgba8888Obj) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to get ARGB_8888 object");
        return NULL;
    }
    
    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    if (!bitmapClass) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find Bitmap class");
        return NULL;
    }
    
    jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    if (!createBitmapMethodID) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find createBitmap method");
        return NULL;
    }
    
    resultBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, 
                                             result_rgba.cols, result_rgba.rows, rgba8888Obj);
    
    if (!resultBitmap) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to create result bitmap");
        return NULL;
    }
    
    // Copy data to the new bitmap
    void* resultPixels;
    if (AndroidBitmap_lockPixels(env, resultBitmap, &resultPixels) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "AndroidBitmap_lockPixels failed for result bitmap");
        return NULL;
    }
    
    memcpy(resultPixels, result_rgba.data, result_rgba.total() * result_rgba.elemSize());
    AndroidBitmap_unlockPixels(env, resultBitmap);
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Detection complete, returning bitmap");
    return resultBitmap;
}

// public native void releaseModel();
JNIEXPORT void JNICALL Java_ncnn_v5lite_demo_ImageDetector_releaseModel(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "releaseModel");
    
    ncnn::MutexLockGuard g(imageLock);
    delete g_imageDetector;
    g_imageDetector = 0;
}

} // extern "C"
