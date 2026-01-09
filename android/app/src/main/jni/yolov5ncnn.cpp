// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolov5.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb, double t1)
{
    double t = t1;
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "TIME:%.2f FPS=%.2f", t, avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static Yolov5* g_yolov5 = 0;
static ncnn::Mutex lock;
static bool g_isTakingPhoto = false;
static cv::Mat g_photoImage;
static bool g_hasPhotoResult = false;
static ncnn::Mutex photoLock; // 添加互斥锁保护照片相关变量

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    bool isTakingPhotoLocal = false;
    bool hasPhotoResultLocal = false;
    cv::Mat photoImageLocal;
    
    // 首先读取全局变量的值，避免在处理过程中被其他线程修改
    { 
        ncnn::MutexLockGuard g(photoLock);
        isTakingPhotoLocal = g_isTakingPhoto;
        hasPhotoResultLocal = g_hasPhotoResult;
        if (hasPhotoResultLocal && !g_photoImage.empty()) {
            photoImageLocal = g_photoImage.clone();
        }
    }
    
    if (isTakingPhotoLocal) { 
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Taking photo now");
        
        // 拍照模式，保存当前帧
        cv::Mat capturedFrame = rgb.clone();
        
        if (capturedFrame.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to capture frame");
        } else {
            // 在拍照图像上进行识别
            std::vector<Object> objects;
            double t = 0;
            bool detectionSuccess = false;
            
            { 
                ncnn::MutexLockGuard g(lock);
                if (g_yolov5) { 
                    t = g_yolov5->detect(capturedFrame, objects);
                    g_yolov5->draw(capturedFrame, objects);
                    detectionSuccess = true;
                }
            }
            
            // 更新全局状态和结果
            if (detectionSuccess) { 
                ncnn::MutexLockGuard g(photoLock);
                g_photoImage = capturedFrame.clone();
                g_hasPhotoResult = true;
                g_isTakingPhoto = false;
                
                // 在预览窗口显示拍照结果
                rgb = capturedFrame.clone();
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Photo taken and processed successfully");
            }
        }
    } else if (hasPhotoResultLocal && !photoImageLocal.empty()) { 
        // 继续显示拍照结果
        rgb = photoImageLocal.clone();
    } else { 
        // 普通实时模式
        double t;
        { 
            ncnn::MutexLockGuard g(lock);
            
            if (g_yolov5) { 
                std::vector<Object> objects;
                t = g_yolov5->detect(rgb, objects);
                g_yolov5->draw(rgb, objects);
            } else { 
                draw_unsupported(rgb);
            }
        }
        
        draw_fps(rgb, t);
    }
}

static MyNdkCamera* g_camera = 0;

extern "C" { 

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    { 
        ncnn::MutexLockGuard g(lock);
        
        delete g_yolov5;
        g_yolov5 = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1) {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

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
        320, // 320-lite-e
        416, // 416-lite-e
        320, // 320-lite-i8e
        416, // 416-lite-i8e
        416, // 416-lite-s
        416, // 416-lite-i8s
        512  // 512-lite-c
    };

    const char* modeltype = modeltypes[(int)modelid];
    int target_size = target_sizes[(int)modelid];
    bool use_gpu = false; // 统一禁用 GPU 强制 CPU

    // reload
    { 
        ncnn::MutexLockGuard g(lock);
        
        if (use_gpu && ncnn::get_gpu_count() == 0) { 
            // no gpu
            delete g_yolov5;
            g_yolov5 = 0;
        } else { 
            if (!g_yolov5) 
                g_yolov5 = new Yolov5;
            g_yolov5->load(mgr, modeltype, target_size, use_gpu);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    // 重置照片状态
    { 
        ncnn::MutexLockGuard g(photoLock);
        g_isTakingPhoto = false;
        g_hasPhotoResult = false;
        g_photoImage.release();
    }
    
    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

// public native boolean takePhoto();
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_takePhoto(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "takePhoto called");
    
    { 
        ncnn::MutexLockGuard g(photoLock);
        g_isTakingPhoto = true;
        g_hasPhotoResult = false;
        if (!g_photoImage.empty()) { 
            g_photoImage.release();
        }
    }
    
    return JNI_TRUE;
}

// Added unified bitmap detection API reusing g_yolov5
// public native Bitmap detectBitmap(Bitmap bitmap);
JNIEXPORT jobject JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_detectBitmap(JNIEnv* env, jobject thiz, jobject bitmap)
{
    if (!bitmap)
        return 0;

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0)
        return 0;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return 0;

    void* pixels = 0;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return 0;

    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);

    std::vector<Object> objects;
    double t = 0;
    {
        ncnn::MutexLockGuard g(lock);
        if (g_yolov5)
        {
            t = g_yolov5->detect(bgr, objects);
            g_yolov5->draw(bgr, objects);
        }
        else
        {
            return 0;
        }
    }

    // overlay time
    char text[64];
    sprintf(text, "Time: %.2f ms", t);
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(bgr, cv::Rect(cv::Point(10, 10), cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255,255,255), -1);
    cv::putText(bgr, text, cv::Point(10, 10 + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0), 1);

    cv::Mat result_rgba;
    cv::cvtColor(bgr, result_rgba, cv::COLOR_BGR2RGBA);

    jclass bitmapClass = env->FindClass("android/graphics/Bitmap");
    jclass configClass = env->FindClass("android/graphics/Bitmap$Config");
    jfieldID argb8888Field = env->GetStaticFieldID(configClass, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    jobject argb8888Obj = env->GetStaticObjectField(configClass, argb8888Field);
    jmethodID createBitmap = env->GetStaticMethodID(bitmapClass, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jobject outBmp = env->CallStaticObjectMethod(bitmapClass, createBitmap, result_rgba.cols, result_rgba.rows, argb8888Obj);
    void* outPixels = 0;
    if (AndroidBitmap_lockPixels(env, outBmp, &outPixels) < 0)
        return 0;
    memcpy(outPixels, result_rgba.data, result_rgba.total()*result_rgba.elemSize());
    AndroidBitmap_unlockPixels(env, outBmp);
    return outBmp;
}

// public native Obj[] detectObjects(Bitmap bitmap);
JNIEXPORT jobjectArray JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_detectObjects(JNIEnv* env, jobject thiz, jobject bitmap)
{
    if (!bitmap)
        return 0;

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0)
        return 0;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return 0;

    void* pixels = 0;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0)
        return 0;

    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    AndroidBitmap_unlockPixels(env, bitmap);

    std::vector<Object> objects;
    {
        ncnn::MutexLockGuard g(lock);
        if (g_yolov5)
        {
            g_yolov5->detect(bgr, objects);
        }
        else
        {
            return 0;
        }
    }

    // Create Java Obj array
    jclass objClass = env->FindClass("ncnn/v5lite/demo/Ncnnv5lite$Obj");
    jobjectArray objArray = env->NewObjectArray(objects.size(), objClass, NULL);

    jmethodID objConstructor = env->GetMethodID(objClass, "<init>", "()V");
    jfieldID xField = env->GetFieldID(objClass, "x", "F");
    jfieldID yField = env->GetFieldID(objClass, "y", "F"); 
    jfieldID wField = env->GetFieldID(objClass, "w", "F");
    jfieldID hField = env->GetFieldID(objClass, "h", "F");
    jfieldID labelField = env->GetFieldID(objClass, "label", "Ljava/lang/String;");
    jfieldID probField = env->GetFieldID(objClass, "prob", "F");

    // Class names for COCO dataset (same as used in yolov5.cpp)
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        jobject obj = env->NewObject(objClass, objConstructor);
        
        const Object& object = objects[i];
        env->SetFloatField(obj, xField, object.rect.x);
        env->SetFloatField(obj, yField, object.rect.y);
        env->SetFloatField(obj, wField, object.rect.width);
        env->SetFloatField(obj, hField, object.rect.height);
        env->SetFloatField(obj, probField, object.prob);
        
        if (object.label >= 0 && object.label < 80)
        {
            jstring labelStr = env->NewStringUTF(class_names[object.label]);
            env->SetObjectField(obj, labelField, labelStr);
        }
        
        env->SetObjectArrayElement(objArray, i, obj);
    }

    return objArray;
}

// public native boolean isModelLoaded();
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_isModelLoaded(JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(lock);
    return g_yolov5 != 0 ? JNI_TRUE : JNI_FALSE;
}


} // extern "C"
