package ncnn.v5lite.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class ImageDetector {
    // JNI methods
    public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
    public native Bitmap detectImage(Bitmap bitmap);
    public native void releaseModel();

    static {
        System.loadLibrary("ncnnv5lite");
    }
}
