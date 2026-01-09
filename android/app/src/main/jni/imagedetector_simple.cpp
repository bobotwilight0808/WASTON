#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
#include <unistd.h>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yolov5.h"
#include <platform.h>
#include <benchmark.h>
#include <cpu.h>

static Yolov5* g_imageDetector = 0;
static ncnn::Mutex imageLock;

extern "C" {

// ImageDetector implementation

JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_ImageDetector_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "loadModel called with modelid=%d, cpugpu=%d", (int)modelid, (int)cpugpu);
    
    // 在模型加载时就设置线程控制，避免在检测时重复设置
    cv::setNumThreads(1);
    
    // 强制禁用所有多线程
    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif
    
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Invalid parameters: modelid=%d, cpugpu=%d", (int)modelid, (int)cpugpu);
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to get AssetManager");
        return JNI_FALSE;
    }

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

    {
        ncnn::MutexLockGuard g(imageLock);
        
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
        
        __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Model loaded successfully: %s", modeltype);
    }

    return JNI_TRUE;
}

JNIEXPORT jobject JNICALL Java_ncnn_v5lite_demo_ImageDetector_detectImage(JNIEnv* env, jobject thiz, jobject bitmap)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "detectImage called");
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Starting model check");
    
    // Check if model is loaded - 暂时简化以定位问题
    if (g_imageDetector == 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Model not loaded");
        return NULL;
    }
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Model check passed");
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Starting bitmap validation");
    
    if (!bitmap) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Input bitmap is null");
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Bitmap is valid, getting bitmap info");
    
    // Get bitmap info
    AndroidBitmapInfo info;
    int ret = AndroidBitmap_getInfo(env, bitmap, &info);
    if (ret < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "AndroidBitmap_getInfo failed, error: %d", ret);
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Bitmap info: %dx%d, format: %d", info.width, info.height, info.format);
    
    // 强制按照ARGB_8888处理，忽略格式检查
    // 因为Java层已经确保了格式转换
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Forcing ARGB_8888 processing regardless of reported format");
    
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
    
    // 强制按ARGB_8888格式创建Mat，因为Java层已保证了格式
    cv::Mat rgba, bgr;
    
    // 创建一个独立的内存拷贝，避免直接使用Android Bitmap内存
    int width = info.width;
    int height = info.height;
    int total_size = width * height * 4;
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Allocating %d bytes for %dx%d image", total_size, width, height);
    
    // 分配独立内存
    unsigned char* local_pixels = new(std::nothrow) unsigned char[total_size];
    if (!local_pixels) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to allocate memory for image copy");
        AndroidBitmap_unlockPixels(env, bitmap);
        return NULL;
    }
    
    // 逐字节安全复制，避免大块内存操作
    unsigned char* src = (unsigned char*)pixels;
    for (int i = 0; i < total_size; i++) {
        local_pixels[i] = src[i];
    }
    
    // 提前解锁bitmap，避免长时间持有
    AndroidBitmap_unlockPixels(env, bitmap);
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Memory copy completed, creating Mat");
    
    // 使用拷贝的数据创建Mat
    rgba = cv::Mat(height, width, CV_8UC4, local_pixels);
    
    // 验证Mat创建是否成功
    if (rgba.empty() || rgba.data == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to create RGBA Mat");
        delete[] local_pixels;
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "RGBA Mat created: %dx%d, channels: %d", rgba.cols, rgba.rows, rgba.channels());
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Starting manual color conversion");
    
    // 手动进行RGBA到BGR转换，避免OpenCV内部的多线程问题
    int mat_width = rgba.cols;
    int mat_height = rgba.rows;
    bgr = cv::Mat(mat_height, mat_width, CV_8UC3);
    
    unsigned char* rgba_data = rgba.data;
    unsigned char* bgr_data = bgr.data;
    
    for (int i = 0; i < mat_height * mat_width; i++) {
        // RGBA to BGR: R->B, G->G, B->R, 跳过A
        bgr_data[i * 3 + 0] = rgba_data[i * 4 + 2]; // B
        bgr_data[i * 3 + 1] = rgba_data[i * 4 + 1]; // G
        bgr_data[i * 3 + 2] = rgba_data[i * 4 + 0]; // R
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Manual color conversion completed");
    
    // 立即释放本地像素内存，减少内存占用
    delete[] local_pixels;
    local_pixels = nullptr;
    
    // 验证转换结果
    if (bgr.empty() || bgr.data == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Color conversion failed");
        return NULL;
    }
    
    if (bgr.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Color conversion failed");
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Color conversion completed");
    
    // Perform detection - 暂时简化以定位问题
    std::vector<Object> objects;
    double t = 0;
    
    if (g_imageDetector) {
        __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Starting detection...");
        
        // 添加更多的稳定性检查
        usleep(5000); // 5ms delay for stability
        
        // Check if bgr is still valid before detection
        if (bgr.empty() || bgr.data == nullptr) {
            __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "BGR Mat is invalid before detection");
            return NULL;
        }
        
        __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "About to call YOLO detect, BGR size: %dx%d", bgr.cols, bgr.rows);
        
        // 使用更保守的检测参数，可能能避免某些复杂计算
        float prob_threshold = 0.8f;  // 更高的阈值，减少计算量
        float nms_threshold = 0.4f;   // 更严格的NMS，减少后处理
        
        t = g_imageDetector->detect(bgr, objects, prob_threshold, nms_threshold);
        __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Detection completed, found %d objects, time: %.2f ms", (int)objects.size(), t);
            
        // Check if bgr is still valid after detection
        if (bgr.empty() || bgr.data == nullptr) {
            __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "BGR Mat became invalid after detection");
            return NULL;
        }
        
        g_imageDetector->draw(bgr, objects);
        __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Drawing completed");
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Model not loaded during detection");
        return NULL;
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
    
    // Convert back to RGBA format
    cv::Mat result_rgba;
    cv::cvtColor(bgr, result_rgba, cv::COLOR_BGR2RGBA);
    
    if (result_rgba.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Result color conversion failed");
        return NULL;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Result color conversion completed");
    
    // Create result bitmap using createBitmap instead of copy
    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    if (!bitmapClass) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to find Bitmap class");
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
    
    // Create new bitmap
    jmethodID createBitmapMethodID = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    if (!createBitmapMethodID) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to find createBitmap method");
        return NULL;
    }
    
    jobject resultBitmap = env->CallStaticObjectMethod(bitmapClass, createBitmapMethodID, 
                                                      result_rgba.cols, result_rgba.rows, config);
    if (!resultBitmap) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to create bitmap");
        return NULL;
    }
    
    // Lock the result bitmap and copy our data
    void* resultPixels;
    ret = AndroidBitmap_lockPixels(env, resultBitmap, &resultPixels);
    if (ret < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "AndroidBitmap_lockPixels failed for result bitmap, error: %d", ret);
        return NULL;
    }
    
    if (!resultPixels) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Result bitmap pixels is null");
        AndroidBitmap_unlockPixels(env, resultBitmap);
        return NULL;
    }
    
    // Get result bitmap info to ensure size matches
    AndroidBitmapInfo resultInfo;
    ret = AndroidBitmap_getInfo(env, resultBitmap, &resultInfo);
    if (ret < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Failed to get result bitmap info");
        AndroidBitmap_unlockPixels(env, resultBitmap);
        return NULL;
    }
    
    // Verify dimensions match
    if (resultInfo.width != (uint32_t)result_rgba.cols || resultInfo.height != (uint32_t)result_rgba.rows) {
        __android_log_print(ANDROID_LOG_ERROR, "ImageDetector", "Dimension mismatch: result bitmap %dx%d vs opencv mat %dx%d", 
                          resultInfo.width, resultInfo.height, result_rgba.cols, result_rgba.rows);
        AndroidBitmap_unlockPixels(env, resultBitmap);
        return NULL;
    }
    
    // Calculate safe copy size
    size_t expectedSize = result_rgba.total() * result_rgba.elemSize();
    size_t bitmapSize = resultInfo.height * resultInfo.stride;
    size_t copySize = (expectedSize < bitmapSize) ? expectedSize : bitmapSize;
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Copying %zu bytes to result bitmap", copySize);
    
    // Copy the processed image data safely
    memcpy(resultPixels, result_rgba.data, copySize);
    
    AndroidBitmap_unlockPixels(env, resultBitmap);
    
    __android_log_print(ANDROID_LOG_DEBUG, "ImageDetector", "Detection complete, returning bitmap");
    return resultBitmap;
}

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
