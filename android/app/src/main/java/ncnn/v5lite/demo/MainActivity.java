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

package ncnn.v5lite.demo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

public class MainActivity extends Activity implements SurfaceHolder.Callback
{
    public static final int REQUEST_CAMERA = 100;

    private final ModelManager modelManager = ModelManager.getInstance();
    private Ncnnv5lite ncnnyolov5 = modelManager.getNcnn();
    private int facing = 1;

    // 移除模型选择器
    // private Spinner spinnerModel;
    // 固定使用 CPU 推理
    private final int current_cpugpu = 0;

    private SurfaceView cameraView;
    private boolean modelReady = false;
    private boolean pendingOpenCamera = false;

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        // Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        // buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
        //     @Override
        //     public void onClick(View arg0) {

        //         int new_facing = 1 - facing;

        //         ncnnyolov5.closeCamera();

        //         ncnnyolov5.openCamera(new_facing);

        //         facing = new_facing;
        //     }
        // });


    // Capture button
        Button buttonTakePhoto = (Button) findViewById(R.id.buttonTakePhoto);
        buttonTakePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                ncnnyolov5.takePhoto();
            }
        });
    // Resume live detection after single capture result
        Button buttonReturnToLive = (Button) findViewById(R.id.buttonReturnToLive);
        buttonReturnToLive.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                // 通过重新打开摄像头来返回实时模式
                ncnnyolov5.closeCamera();
                ncnnyolov5.openCamera(facing);
            }
        });

    // Main menu button
        Button buttonBackToMenu = (Button) findViewById(R.id.buttonBackToMenu);
        if (buttonBackToMenu != null) {
            buttonBackToMenu.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    // 关闭摄像头并返回主菜单
                    ncnnyolov5.closeCamera();
                    Intent intent = new Intent(MainActivity.this, ModeSelectionActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
                    startActivity(intent);
                    finish();
                }
            });
        }

        // 移除模型选择器相关代码
        /*
        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id)
            {
                if (position != current_model)
                {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0)
            {
            }
        });
        */

    // 已移除 CPU/GPU 选择器

    // 不在 onCreate 直接加载，改在 surfaceCreated 中延迟执行防止界面切换瞬间并发
    }

    private void reload() {
        new android.os.Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Log.d("MainActivity", "Calling loadModel...");
                boolean ret_init = modelManager.ensureModelLoaded(getAssets(), 0, current_cpugpu);
                if (!ret_init) {
                    Log.e("MainActivity", "loadModel failed");
                } else {
                    modelReady = true;
                    Log.d("MainActivity", "Model loaded");
                    if (pendingOpenCamera) {
                        pendingOpenCamera = false;
                        openCameraSafe();
                    }
                }
            }
        }, 150);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        ncnnyolov5.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    Log.d("MainActivity", "surfaceCreated, trigger reload");
    reload();
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        } else {
            openCameraSafe();
        }
    }

    @Override
    public void onPause()
    {
        super.onPause();

        ncnnyolov5.closeCamera();
    }

    private void openCameraSafe() {
        if (!modelReady) {
            pendingOpenCamera = true;
            Log.d("MainActivity", "Model not ready, delay openCamera");
            return;
        }
        try {
            Log.d("MainActivity", "Opening camera now");
            ncnnyolov5.openCamera(facing);
        } catch (Throwable t) {
            Log.e("MainActivity", "openCamera error: " + t.getMessage());
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCameraSafe();
            } else {
                Log.e("MainActivity", "Camera permission denied");
            }
        }
    }
}