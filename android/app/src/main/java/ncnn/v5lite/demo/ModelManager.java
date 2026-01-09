package ncnn.v5lite.demo;

import android.content.res.AssetManager;
import android.util.Log;

/**
 * 全局模型单例管理，保证 ncnn 模型在应用进程内只加载一次。
 */
public class ModelManager {
    private static final String TAG = "ModelManager";
    private static volatile ModelManager instance;

    private final Ncnnv5lite ncnn = new Ncnnv5lite();
    private boolean loaded = false;
    private int lastModelId = -1;
    private int lastCpuGpu = -1;

    private ModelManager() {}

    public static ModelManager getInstance() {
        if (instance == null) {
            synchronized (ModelManager.class) {
                if (instance == null) instance = new ModelManager();
            }
        }
        return instance;
    }

    public synchronized boolean ensureModelLoaded(AssetManager assets, int modelId, int cpuGpu) {
        if (loaded && modelId == lastModelId && cpuGpu == lastCpuGpu && ncnn.isModelLoaded()) {
            return true; // 已加载且参数一致
        }
        Log.d(TAG, "Loading model modelId=" + modelId + " cpuGpu=" + cpuGpu);
        boolean ok = false;
        try {
            ok = ncnn.loadModel(assets, modelId, cpuGpu);
        } catch (Throwable t) {
            Log.e(TAG, "loadModel exception", t);
            ok = false;
        }
        if (ok) {
            loaded = true;
            lastModelId = modelId;
            lastCpuGpu = cpuGpu;
        }
        return ok;
    }

    public Ncnnv5lite getNcnn() { return ncnn; }

    public synchronized boolean isLoaded() { return loaded && ncnn.isModelLoaded(); }
}
