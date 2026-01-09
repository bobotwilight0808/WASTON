package ncnn.v5lite.demo;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.PointF;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.View;
// 移除 CPU/GPU 选择器相关 import
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class ImageDetectionActivity extends Activity {
    private static final String TAG = "ImageDetectionActivity";
    private static final int REQUEST_PICK_IMAGE = 1;
    private static final int REQUEST_STORAGE_PERMISSION = 2;

    private ImageView imageViewResult;
    private Button buttonSelectImage;
    private TextView resultTextView;

    private Button buttonSaveResult;
    private Button buttonBack;
    private ProgressBar progressBar;
    // 已移除 CPU/GPU 选择器，固定使用 CPU

    private int current_cpugpu = 0;
    // 复用全局单例模型
    private final ModelManager modelManager = ModelManager.getInstance();
    private final Ncnnv5lite sharedDetector = modelManager.getNcnn();
    private Uri selectedImageUri = null;
    private Bitmap resultBitmap = null;

    // Define 24 preset points in normalized coordinates (0.0 to 1.0)
    // This is an example 6x4 grid. You can change these values.
    private static final List<PointF> presetPoints = new ArrayList<>();
    static {
    // Image dimensions for normalization
        final float imageWidth = 2592f;
        final float imageHeight = 1944f;

        // Add the 24 preset points with normalized coordinates
        presetPoints.add(new PointF( 715f / imageWidth, 1581f / imageHeight));
        presetPoints.add(new PointF( 897f / imageWidth, 1665f / imageHeight));
        presetPoints.add(new PointF(1087f / imageWidth, 1699f / imageHeight));
        presetPoints.add(new PointF(1296f / imageWidth, 1720f / imageHeight));
        presetPoints.add(new PointF(1956f / imageWidth, 1612f / imageHeight));
        presetPoints.add(new PointF(2113f / imageWidth, 1511f / imageHeight));
        presetPoints.add(new PointF(2253f / imageWidth, 1406f / imageHeight));
        presetPoints.add(new PointF(2369f / imageWidth, 1276f / imageHeight));
        presetPoints.add(new PointF(2461f / imageWidth,  835f / imageHeight));
        presetPoints.add(new PointF(2429f / imageWidth,  713f / imageHeight));
        presetPoints.add(new PointF(2359f / imageWidth,  593f / imageHeight));
        presetPoints.add(new PointF(2268f / imageWidth,  482f / imageHeight));
        presetPoints.add(new PointF(1858f / imageWidth,  219f / imageHeight));
        presetPoints.add(new PointF(1704f / imageWidth,  170f / imageHeight));
        presetPoints.add(new PointF(1547f / imageWidth,  148f / imageHeight));
        presetPoints.add(new PointF(1386f / imageWidth,  139f / imageHeight));
        presetPoints.add(new PointF( 872f / imageWidth,  212f / imageHeight));
        presetPoints.add(new PointF( 732f / imageWidth,  270f / imageHeight));
        presetPoints.add(new PointF( 599f / imageWidth,  343f / imageHeight));
        presetPoints.add(new PointF( 489f / imageWidth,  446f / imageHeight));
        presetPoints.add(new PointF( 262f / imageWidth,  807f / imageHeight));
        presetPoints.add(new PointF( 252f / imageWidth,  942f / imageHeight));
        presetPoints.add(new PointF( 269f / imageWidth, 1071f / imageHeight));
        presetPoints.add(new PointF( 313f / imageWidth, 1204f / imageHeight));
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_detection);

        imageViewResult = (ImageView) findViewById(R.id.imageViewResult);
        buttonSelectImage = (Button) findViewById(R.id.buttonSelectImage);
        buttonSaveResult = (Button) findViewById(R.id.buttonSaveResult);
        buttonBack = (Button) findViewById(R.id.buttonBack);
        progressBar = (ProgressBar) findViewById(R.id.progressBar);
        resultTextView = (TextView) findViewById(R.id.resultTextView);

        buttonSelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });

        buttonSaveResult.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (resultBitmap != null) {
                    checkStoragePermissionAndSave();
                } else {
                    Toast.makeText(ImageDetectionActivity.this, getString(R.string.msg_no_result_to_save), Toast.LENGTH_SHORT).show();
                }
            }
        });

        buttonBack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finish(); // 返回到模式选择界面
            }
        });

    // 无需设置 spinner 监听

        // 如果模型尚未加载则加载一次
    modelManager.ensureModelLoaded(getAssets(), 0, current_cpugpu);
    }

    private void loadModel() {
    modelManager.ensureModelLoaded(getAssets(), 0, current_cpugpu);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_PICK_IMAGE);
    }

    private void detectImage() {
        if (selectedImageUri == null) {
            return;
        }

        try {
            progressBar.setVisibility(View.VISIBLE);

            // 在后台线程中进行检测，避免阻塞UI
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        // 获取选择的图片
                        InputStream inputStream = getContentResolver().openInputStream(selectedImageUri);
                        
                        // 添加图片尺寸限制，避免内存问题
                        BitmapFactory.Options options = new BitmapFactory.Options();
                        options.inJustDecodeBounds = true;
                        BitmapFactory.decodeStream(inputStream, null, options);
                        inputStream.close();
                        
                        // 计算采样率，限制最大尺寸为2048x2048
                        int maxSize = 2048;
                        int sampleSize = 1;
                        if (options.outHeight > maxSize || options.outWidth > maxSize) {
                            final int halfHeight = options.outHeight / 2;
                            final int halfWidth = options.outWidth / 2;
                            while ((halfHeight / sampleSize) >= maxSize && (halfWidth / sampleSize) >= maxSize) {
                                sampleSize *= 2;
                            }
                        }
                        
                        Log.d(TAG, "Original image size: " + options.outWidth + "x" + options.outHeight);
                        Log.d(TAG, "Sample size: " + sampleSize);
                        
                        // 重新打开输入流进行实际解码
                        inputStream = getContentResolver().openInputStream(selectedImageUri);
                        
                        // 强制设置为ARGB_8888格式并应用采样
                        options.inJustDecodeBounds = false;
                        options.inSampleSize = sampleSize;
                        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                        options.inMutable = false;
                        options.inDither = false;
                        options.inScaled = false;
                        
                        Bitmap tempBitmap = BitmapFactory.decodeStream(inputStream, null, options);
                        inputStream.close();
                        
                        // 强制转换为ARGB_8888格式
                        Bitmap originalBitmap = null;
                        if (tempBitmap != null) {
                            Log.d(TAG, "Loaded bitmap config: " + tempBitmap.getConfig());
                            Log.d(TAG, "Loaded bitmap size: " + tempBitmap.getWidth() + "x" + tempBitmap.getHeight());
                            
                            // 无论原始格式是什么，都创建一个新的ARGB_8888 bitmap
                            originalBitmap = Bitmap.createBitmap(tempBitmap.getWidth(), tempBitmap.getHeight(), Bitmap.Config.ARGB_8888);
                            Canvas canvas = new Canvas(originalBitmap);
                            canvas.drawBitmap(tempBitmap, 0, 0, null);
                            
                            // 验证最终格式
                            Log.d(TAG, "Final bitmap config: " + originalBitmap.getConfig());
                            Log.d(TAG, "Final bitmap size: " + originalBitmap.getWidth() + "x" + originalBitmap.getHeight());
                            
                            // 回收临时bitmap
                            if (tempBitmap != originalBitmap) {
                                tempBitmap.recycle();
                            }
                        }
                        
                        final Bitmap finalOriginalBitmap = originalBitmap;

                        if (finalOriginalBitmap == null) {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(ImageDetectionActivity.this, getString(R.string.msg_failed_load_image), Toast.LENGTH_SHORT).show();
                                    progressBar.setVisibility(View.GONE);
                                }
                            });
                            return;
                        }

                        Log.d(TAG, "Original bitmap size: " + finalOriginalBitmap.getWidth() + "x" + finalOriginalBitmap.getHeight());

                        // 执行图片检测
                        Bitmap detectionResult = null;
                        Ncnnv5lite.Obj[] detectedObjects = null;
                        String matchResults = "";
                        
                        try {
                            // Get detection results as objects array
                            detectedObjects = sharedDetector.detectObjects(finalOriginalBitmap);
                            // Get rendered bitmap with bounding boxes
                            detectionResult = sharedDetector.detectBitmap(finalOriginalBitmap);
                            
                            // Perform matching logic
                            matchResults = performMatching(detectedObjects, finalOriginalBitmap.getWidth(), finalOriginalBitmap.getHeight());
                            
                            Log.d(TAG, "Detection completed, objects: " + (detectedObjects != null ? detectedObjects.length : 0));
                            Log.d(TAG, "Match results: " + matchResults);
                        } catch (Exception e) {
                            Log.e(TAG, "Detection failed with exception: " + e.getMessage());
                        }
                        
                        final Bitmap finalDetectionResult = detectionResult;
                        final String finalMatchResults = matchResults;

                        // 在UI线程中更新界面
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                if (finalDetectionResult != null) {
                                    resultBitmap = finalDetectionResult;
                                    imageViewResult.setImageBitmap(resultBitmap);
                                    buttonSaveResult.setEnabled(true);
                                    
                                    // Update result text
                                    resultTextView.setText(finalMatchResults);
                                    
                                    Toast.makeText(ImageDetectionActivity.this, getString(R.string.msg_detection_finished), Toast.LENGTH_SHORT).show();
                                } else {
                                    Toast.makeText(ImageDetectionActivity.this, getString(R.string.msg_detection_failed), Toast.LENGTH_SHORT).show();
                                }
                                progressBar.setVisibility(View.GONE);
                            }
                        });
                    } catch (Exception e) {
                        final String errorMessage = e.getMessage();
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(ImageDetectionActivity.this, getString(R.string.msg_detection_error, errorMessage), Toast.LENGTH_SHORT).show();
                                progressBar.setVisibility(View.GONE);
                            }
                        });
                    }
                }
            }).start();
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.msg_processing_error, e.getMessage()), Toast.LENGTH_SHORT).show();
            progressBar.setVisibility(View.GONE);
        }
    }

    private void checkStoragePermissionAndSave() {
        // 检查存储权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    REQUEST_STORAGE_PERMISSION);
        } else {
            saveResultToGallery();
        }
    }

    private void saveResultToGallery() {
        if (resultBitmap == null) {
            Toast.makeText(this, getString(R.string.msg_no_result_to_save), Toast.LENGTH_SHORT).show();
            return;
        }

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "YOLO_DETECTION_" + timeStamp;

        try {
            // 使用传统方式保存图片
            String path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).toString() + File.separator + "YOLODetections";
            File directory = new File(path);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            File file = new File(directory, imageFileName + ".jpg");
            FileOutputStream outputStream = new FileOutputStream(file);
            resultBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
            outputStream.flush();
            outputStream.close();

            // 通知媒体库更新
            Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
            Uri contentUri = Uri.fromFile(file);
            mediaScanIntent.setData(contentUri);
            this.sendBroadcast(mediaScanIntent);

            Toast.makeText(this, getString(R.string.msg_result_saved), Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Log.e(TAG, "保存图片失败: " + e.getMessage());
            Toast.makeText(this, getString(R.string.msg_save_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            selectedImageUri = data.getData();
            try {
                InputStream inputStream = getContentResolver().openInputStream(selectedImageUri);
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                imageViewResult.setImageBitmap(bitmap);
                inputStream.close();
                buttonSaveResult.setEnabled(false);  // 重置保存按钮状态
                resultBitmap = null;  // 清除之前的检测结果
                detectImage();
            } catch (Exception e) {
                Toast.makeText(this, getString(R.string.msg_failed_load_image), Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_STORAGE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                saveResultToGallery();
            } else {
                Toast.makeText(this, getString(R.string.msg_storage_permission), Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 不释放 sharedDetector 模型，避免影响实时检测
    }

    private String performMatching(Ncnnv5lite.Obj[] detectedObjects, int imageWidth, int imageHeight) {
        if (presetPoints.isEmpty()) {
            return "No preset points defined.";
        }

        // This threshold determines how close a detection center must be to a preset point to be a match.
        // Here, it's set to 10% of the image's width. You can adjust this value.
        final double distanceThreshold = imageWidth * 0.1;

        String[] pointStatus = new String[presetPoints.size()];

        // Initialize all points as "negative"
        for (int i = 0; i < presetPoints.size(); i++) {
            pointStatus[i] = "negative";
        }

        if (detectedObjects != null && detectedObjects.length > 0) {
            // For each detected object, find the closest preset point and mark it as "positive"
            for (Ncnnv5lite.Obj obj : detectedObjects) {
                float centerX = obj.x + obj.w / 2;
                float centerY = obj.y + obj.h / 2;

                double minDistance = Double.MAX_VALUE;
                int closestPointIndex = -1;

                for (int i = 0; i < presetPoints.size(); i++) {
                    PointF preset = presetPoints.get(i);
                    // Convert normalized preset point to image coordinates
                    float presetX = preset.x * imageWidth;
                    float presetY = preset.y * imageHeight;

                    double distance = Math.sqrt(Math.pow(centerX - presetX, 2) + Math.pow(centerY - presetY, 2));

                    if (distance < minDistance) {
                        minDistance = distance;
                        closestPointIndex = i;
                    }
                }

                // If the closest point is within the threshold, mark it as positive
                if (closestPointIndex != -1 && minDistance < distanceThreshold) {
                    pointStatus[closestPointIndex] = "positive";
                }
            }
        }

        // Selected point indices: 1,2,3, 5,6,7, 9,10,11, 13,14,15, 17,18,19, 21,22,23
        // (0-based indexing: 0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18, 20,21,22)
        int[][] selectedPoints = {
            {0, 1, 2},      // Row 1: points 1,2,3
            {4, 5, 6},      // Row 2: points 5,6,7
            {8, 9, 10},     // Row 3: points 9,10,11
            {12, 13, 14},   // Row 4: points 13,14,15
            {16, 17, 18},   // Row 5: points 17,18,19
            {20, 21, 22}    // Row 6: points 21,22,23
        };

        // Row IDs for display
        String[] rowIds = {"1", "2", "3", "4", "5", "6"};

        // Build table format with headers like the image
        StringBuilder resultsBuilder = new StringBuilder();
        
        // Header row - column widths: ID=10, miR-17=12, miR-155=12, miR-19b=12, Result=20
        resultsBuilder.append(String.format("%-10s%-12s%-12s%-15s%-18s\n", 
            "ID", "miR-17", "miR-155", "miR-19b", "Result"));
        resultsBuilder.append("\n");

        for (int row = 0; row < 6; row++) {
            // Three columns of detection results
            int positiveCount = 0;
            String[] symbols = new String[3];
            
            for (int col = 0; col < 3; col++) {
                int pointIndex = selectedPoints[row][col];
                String status = pointStatus[pointIndex];
                
                if (status.equals("positive")) {
                    positiveCount++;
                }
                
                // Display + or - instead of positive/negative
                symbols[col] = status.equals("positive") ? "+" : " -";
            }
            
            // Result column: count positives and negatives
            int negativeCount = 3 - positiveCount;
            String result;
            if (positiveCount == 0) {
                result = "     3 -";
            } else if (negativeCount == 0) {
                result = "     3 +";
            } else {
                result = positiveCount + " +"  +
                         ", " + negativeCount + " -";
            }
            
            // Format data row with SAME widths as header: 10, 12, 12, 12, 20
            resultsBuilder.append(String.format("%-16s%-19s%-21s%-16s%-20s\n", 
                rowIds[row], symbols[0], symbols[1], symbols[2], result));

            // Increased line spacing
            resultsBuilder.append("\n");
        }

        return resultsBuilder.toString();
    }



}
