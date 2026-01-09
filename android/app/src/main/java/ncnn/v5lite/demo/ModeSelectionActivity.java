package ncnn.v5lite.demo;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class ModeSelectionActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mode_selection);

        Button buttonRealTime = (Button) findViewById(R.id.buttonRealTime);
        Button buttonImageUpload = (Button) findViewById(R.id.buttonImageUpload);

        // 实时检测按钮
        buttonRealTime.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ModeSelectionActivity.this, MainActivity.class);
                startActivity(intent);
            }
        });

        // 图片检测按钮
        buttonImageUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ModeSelectionActivity.this, ImageDetectionActivity.class);
                startActivity(intent);
            }
        });
    }
}
