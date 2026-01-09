// This is an improved version of specific methods for handling image upload and display
// Add these methods to your existing yolov5ncnn.cpp file

// Enhanced version of setOutputWindow to better handle image results
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    // First check if there is a photo result to display
    bool hasPhotoToDisplay = false;
    cv::Mat photoToDisplay;
    
    {
        ncnn::MutexLockGuard g(photoLock);
        hasPhotoToDisplay = g_hasPhotoResult && !g_photoImage.empty();
        if (hasPhotoToDisplay) {
            photoToDisplay = g_photoImage.clone();
        }
    }
    
    if (hasPhotoToDisplay) {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "直接渲染图片结果, 大小: %dx%d", 
            photoToDisplay.cols, photoToDisplay.rows);
            
        // Directly render the image result to the surface
        if (win != NULL) {
            // Configure buffer geometry to match image dimensions
            ANativeWindow_setBuffersGeometry(win, photoToDisplay.cols, photoToDisplay.rows, 
                                           AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM);
            
            ANativeWindow_Buffer buf;
            if (ANativeWindow_lock(win, &buf, NULL) == 0) {
                // Copy image data to the buffer
                if (buf.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM || 
                    buf.format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) {
                    
                    for (int y = 0; y < photoToDisplay.rows && y < buf.height; y++) {
                        const unsigned char* ptr = photoToDisplay.ptr<const unsigned char>(y);
                        unsigned char* outptr = (unsigned char*)buf.bits + buf.stride * 4 * y;
                        
                        for (int x = 0; x < photoToDisplay.cols && x < buf.width; x++) {
                            outptr[0] = ptr[0]; // B
                            outptr[1] = ptr[1]; // G
                            outptr[2] = ptr[2]; // R
                            outptr[3] = 255;    // A
                            
                            ptr += 3;
                            outptr += 4;
                        }
                    }
                    
                    ANativeWindow_unlockAndPost(win);
                    return JNI_TRUE;
                } else {
                    ANativeWindow_unlockAndPost(win);
                }
            }
        }
    }
    
    // If no photo to display or failed to display, set up camera window as usual
    g_camera->set_window(win);
    return JNI_TRUE;
}

// Enhanced version of detectImageBitmap for better handling
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_detectImageBitmap(JNIEnv* env, jobject thiz, jobject bitmap)
{
    if (g_yolov5 == 0)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "detectImageBitmap called");
    
    // Get bitmap info
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "AndroidBitmap_getInfo failed");
        return JNI_FALSE;
    }
    
    // Check bitmap format
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Bitmap format is not RGBA_8888");
        return JNI_FALSE;
    }
    
    // Lock bitmap pixels
    void* pixels;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "AndroidBitmap_lockPixels failed");
        return JNI_FALSE;
    }
    
    // Create OpenCV Mat from bitmap
    cv::Mat rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat bgr(info.height, info.width, CV_8UC3);
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    
    // Unlock bitmap pixels
    AndroidBitmap_unlockPixels(env, bitmap);
    
    // Make sure camera is closed
    g_camera->close();
    
    // Perform detection
    std::vector<Object> objects;
    double t = 0;
    bool detectionSuccess = false;
    
    {
        ncnn::MutexLockGuard g(lock);
        if (g_yolov5) {
            t = g_yolov5->detect(bgr, objects);
            g_yolov5->draw(bgr, objects);
            
            // Add a text marker indicating this is from gallery
            char text[32];
            sprintf(text, "From Gallery");
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseLine);
            cv::rectangle(bgr, cv::Rect(cv::Point(20, 20), 
                        cv::Size(label_size.width, label_size.height + baseLine)),
                        cv::Scalar(255, 255, 255), -1);
            cv::putText(bgr, text, cv::Point(20, 20 + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
                        
            detectionSuccess = true;
        }
    }
    
    // Save processed image and update state
    if (detectionSuccess) {
        ncnn::MutexLockGuard g(photoLock);
        // Clear previous image to prevent memory leaks
        if (!g_photoImage.empty()) {
            g_photoImage.release();
        }
        g_photoImage = bgr.clone();
        g_hasPhotoResult = true;
        g_isTakingPhoto = false;
        
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Image detected successfully - size: %dx%d", 
                           g_photoImage.cols, g_photoImage.rows);
        
        // Verify the image isn't empty
        if (g_photoImage.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "ERROR: Processed image is empty!");
            return JNI_FALSE;
        }
    }
    
    return detectionSuccess ? JNI_TRUE : JNI_FALSE;
}

// Add a new method to directly render a bitmap to a surface without going through the camera
JNIEXPORT jboolean JNICALL Java_ncnn_v5lite_demo_Ncnnv5lite_directRenderImage(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    
    if (win == NULL) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "directRenderImage: Invalid surface");
        return JNI_FALSE;
    }
    
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "directRenderImage: Rendering to surface %p", win);
    
    bool success = false;
    
    // Make a copy of the photo to render
    cv::Mat photoToRender;
    {
        ncnn::MutexLockGuard g(photoLock);
        if (g_hasPhotoResult && !g_photoImage.empty()) {
            photoToRender = g_photoImage.clone();
            success = true;
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "directRenderImage: No photo available to render");
        }
    }
    
    if (success) {
        // Set the buffer geometry to match the image dimensions
        ANativeWindow_setBuffersGeometry(win, photoToRender.cols, photoToRender.rows, 
                                       AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM);
        
        // Lock the buffer
        ANativeWindow_Buffer buf;
        int lockResult = ANativeWindow_lock(win, &buf, NULL);
        
        if (lockResult == 0) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "directRenderImage: Buffer locked, format=%d", buf.format);
            
            // Check if the buffer format is compatible
            if (buf.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM || 
                buf.format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) {
                
                // Copy the image data to the buffer
                for (int y = 0; y < photoToRender.rows && y < buf.height; y++) {
                    const unsigned char* ptr = photoToRender.ptr<const unsigned char>(y);
                    unsigned char* outptr = (unsigned char*)buf.bits + buf.stride * 4 * y;
                    
                    for (int x = 0; x < photoToRender.cols && x < buf.width; x++) {
                        outptr[0] = ptr[0]; // B
                        outptr[1] = ptr[1]; // G
                        outptr[2] = ptr[2]; // R
                        outptr[3] = 255;    // A
                        
                        ptr += 3;
                        outptr += 4;
                    }
                }
                
                ANativeWindow_unlockAndPost(win);
                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "directRenderImage: Successfully rendered image");
                return JNI_TRUE;
            } else {
                __android_log_print(ANDROID_LOG_ERROR, "ncnn", "directRenderImage: Unsupported buffer format: %d", buf.format);
                ANativeWindow_unlockAndPost(win);
            }
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "directRenderImage: Failed to lock buffer, error: %d", lockResult);
        }
    }
    
    return JNI_FALSE;
}
