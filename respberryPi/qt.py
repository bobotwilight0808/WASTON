import cv2
import numpy as np
import onnxruntime as ort
import time
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QHBoxLayout, QFileDialog, QTextEdit, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import sys
from picamera2 import Picamera2
import os
import matplotlib
os.environ['DISPLAY'] = ':0'  


class YOLOv5LiteApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Waston')
        self.setGeometry(100, 100, 1000, 800)
        self.crop_x_ratio = 0.31
        self.crop_y_ratio = 0.39
        self.crop_w_ratio = 0.38
        self.crop_h_ratio = 0.49
        self.picam2 = Picamera2()
        self.video_config = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 960)})
        self.still_config = self.picam2.create_still_configuration()
        self.cap = None
        self.net = None
        self.is_camera_active = False
        self.last_image = None
        self.detection_active = False
        self.init_ui()
        self.ref_points = np.array([[ 715, 1581],
                            [ 897, 1665],
                            [1087, 1699],
                            [1296, 1720],
                            [1956, 1612],
                            [2113, 1511],
                            [2253, 1406],
                            [2369, 1276],
                            [2461,  835],
                            [2429,  713],
                            [2359,  593],
                            [2268,  482],
                            [1858,  219],
                            [1704,  170],
                            [1547,  148],
                            [1386,  139],
                            [ 872,  212],
                            [ 732,  270],
                            [ 599,  343],
                            [ 489,  446],
                            [ 262,  807],
                            [ 252,  942],
                            [ 269, 1071],
                            [ 313, 1204]])
      
        try:
            self.load_model()
        except Exception as e:
            self.status_label.setText(f'模型加载失败: {str(e)}')

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
     
        self.display_image()
    def init_ui(self):
  
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        
        self.image_label = QLabel('请点击开始摄像头')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet('background-color: #f0f0f0; border: 1px solid #ccc;')
        self.image_label.setFixedSize(460, 460)
        main_layout.addWidget(self.image_label)
        
        button_layout = QHBoxLayout()
        
       
        self.camera_button = QPushButton('开始摄像头')
        self.camera_button.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.camera_button)
        
      
        self.capture_button = QPushButton('拍照')
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)
        button_layout.addWidget(self.capture_button)
        
    
        self.detect_button = QPushButton('开启检测')
        self.detect_button.clicked.connect(self.toggle_detection)
        self.detect_button.setEnabled(False)
        button_layout.addWidget(self.detect_button)
        
        
        self.load_button = QPushButton('加载图片')
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)
        
        main_layout.addLayout(button_layout)
        
        #文本显示区域
        self.status_label = QTextEdit()
        self.status_label.setReadOnly(True)  # 设置为只读
        self.status_label.setStyleSheet('background-color: #f0f0f0; border: 1px solid #ccc;')
        main_layout.addWidget(self.status_label)

        # Update QTableWidget initialization to remove ID column
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["miR-17", "miR-155", "miR-19b", "result"])
        self.result_table.setStyleSheet('background-color: #f0f0f0; border: 1px solid #ccc;')
        self.result_table.hide()  # Default to hidden
        main_layout.addWidget(self.result_table)

    
    def load_model(self):
        
        try:
            model_pb_path = "best.onnx"
            so = ort.SessionOptions()
            self.net = ort.InferenceSession(model_pb_path, so)
            
           
            self.model_h = 320
            self.model_w = 320
            self.nl = 3
            self.na = 3
            self.stride = [8., 16., 32.]
            self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
            self.anchor_grid = np.asarray(self.anchors, dtype=np.float32).reshape(self.nl, -1, 2)
            
            self.dic_labels = {0: 'positive'}
            
            self.status_label.setText('模型加载成功')
           
            if self.last_image is not None:
                self.detect_button.setEnabled(True)
        except Exception as e:
            raise e
    
    def toggle_camera(self):
        if not self.is_camera_active:
            try:
                # Initialize Picamera2
                self.picam2.configure(self.video_config)
                full_res = self.picam2.sensor_resolution
                crop_x = int(full_res[0] * self.crop_x_ratio)
                crop_y = int(full_res[1] * self.crop_y_ratio)
                crop_w = int(full_res[0] * self.crop_w_ratio)
                crop_h = int(full_res[1] * self.crop_h_ratio)
                self.picam2.set_controls({"ScalerCrop": (crop_x, crop_y, crop_w, crop_h)})
                self.picam2.start()
                self.status_label.show()
                self.result_table.hide()

            except Exception as e:
                self.status_label.setText(f'摄像头初始化失败: {str(e)}')
                return
            # self.cap = cv2.VideoCapture(0)  
            # if not self.cap.isOpened():
            #     self.status_label.setText('无法打开摄像头')
            #     return
            
            self.is_camera_active = True
            self.camera_button.setText('停止摄像头')
            self.capture_button.setEnabled(True)
          
            self.detect_button.setEnabled(True)
            self.timer.start(30)  
            self.status_label.setText('摄像头已启动')
        else:
            
            self.timer.stop()
            #self.cap.release()
            self.picam2.stop()
            #self.picam2.close()
            self.is_camera_active = False
            self.camera_button.setText('开始摄像头')
            self.capture_button.setEnabled(False)
           
            if self.last_image is None:
                self.detect_button.setEnabled(False)
            self.image_label.setText('请点击开始摄像头')
            self.status_label.setText('摄像头已停止')
    
    def toggle_detection(self):
        self.detection_active = not self.detection_active
        self.detect_button.setText('关闭检测' if self.detection_active else '开启检测')
        self.status_label.setText('目标检测已' + ('开启' if self.detection_active else '关闭'))
        if self.last_image is not None and not self.is_camera_active:
            self.display_image()

    def display_image(self, img=None):
        boxes = None
        working_img = None
        
        if self.last_image is not None:
            working_img = self.last_image.copy()
            if self.detection_active and self.net is not None:
                working_img, boxes = self.perform_detection(working_img)
        else:
            width, height = 640, 640
            working_img = np.zeros((height, width, 3), dtype=np.uint8) + 240
        
        result = working_img.copy()
        rgb_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        if not hasattr(self, 'fixed_display_size'):
        
            self.fixed_display_size = self.image_label.size()

        scaled_pixmap = pixmap.scaled(
            self.fixed_display_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        if not self.is_camera_active and boxes is not None:
            self.check_reference_points(boxes)
    
    def update_frame(self):
        frame = self.picam2.capture_array() 
        # ret, frame = self.cap.read()
        # if not ret:
        #     return
        self.last_image = frame.copy()

      
        if self.detection_active and self.net is not None:
            frame,_ = self.perform_detection(frame)
        
     
        self.display_image(frame)
    
    def perform_detection(self, frame):
    
        det_boxes, scores, ids = self.infer_img(frame)
        
     
        for box, score, id in zip(det_boxes, scores, ids):
            label = '%s:%.2f' % (self.dic_labels[id], score)
            self.plot_one_box(box.astype(np.int16), frame, color=(255, 0, 0), label=label)

        return frame, det_boxes

    def infer_img(self, img0):
    
        img = cv2.resize(img0, [self.model_w, self.model_h], interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        
    
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
        
      
        outs = self.cal_outputs(outs)
        
    
        img_h, img_w, _ = np.shape(img0)
        boxes, confs, ids = self.post_process_opencv(outs, img_h, img_w)
        
        return boxes, confs, ids
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = (line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    
    def cal_outputs(self, outs):
        row_ind = 0
        grid = [np.zeros(1)] * self.nl
        for i in range(self.nl):
            h, w = int(self.model_w / self.stride[i]), int(self.model_h / self.stride[i])
            length = int(self.na * h * w)
            if grid[i].shape[2:4] != (h, w):
                grid[i] = self._make_grid(w, h)
            
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
                grid[i], (self.na, 1))) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
                self.anchor_grid[i], h * w, axis=0)
            row_ind += length
        return outs
    
    def post_process_opencv(self, outputs, img_h, img_w, thred_nms=0.7, thred_cond=0.4):
        conf = outputs[:, 4].tolist()
        c_x = outputs[:, 0] / self.model_w * img_w
        c_y = outputs[:, 1] / self.model_h * img_h
        w = outputs[:, 2] / self.model_w * img_w
        h = outputs[:, 3] / self.model_h * img_h
        p_cls = outputs[:, 5:]
        if len(p_cls.shape) == 1:
            p_cls = np.expand_dims(p_cls, 1)
        cls_id = np.argmax(p_cls, axis=1)
        
        p_x1 = np.expand_dims(c_x - w / 2, -1)
        p_y1 = np.expand_dims(c_y - h / 2, -1)
        p_x2 = np.expand_dims(c_x + w / 2, -1)
        p_y2 = np.expand_dims(c_y + h / 2, -1)
        areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)
        
        areas = areas.tolist()
        ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
        if len(ids) > 0:
            return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
        else:
            return [], [], []
    
    # def capture_image(self):
    #     if self.last_image is not None:
         
    #         timestamp = time.strftime("%Y%m%d_%H%M%S")
    #         filename = f"captured_{timestamp}.jpg"
    #         cv2.imwrite(filename, self.last_image)
    #         self.status_label.setText(f'图片已保存: {filename}')
    def capture_image(self):
        if self.is_camera_active:
            try:
                self.timer.stop()
                self.picam2.stop()

                # 配置摄像头为全分辨率模式
                self.picam2.configure(self.still_config)
                                # 设置裁剪参数（与视频流裁剪一致）
                full_res = self.picam2.sensor_resolution
                crop_x = int(full_res[0] * self.crop_x_ratio)
                crop_y = int(full_res[1] * self.crop_y_ratio)
                crop_w = int(full_res[0] * self.crop_w_ratio)
                crop_h = int(full_res[1] * self.crop_h_ratio)
                self.picam2.set_controls({"ScalerCrop": (crop_x, crop_y, crop_w, crop_h)})
                self.picam2.start()

                # 使用全分辨率拍摄一张图片
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"/home/pi/Desktop/figures/captured_{timestamp}.jpg"
                self.picam2.capture_file(filename)  # 保存全分辨率图片到文件
                self.status_label.setText(f'图片已保存: {filename}')
                self.picam2.stop()
                 # 恢复视频流配置
                self.picam2.configure(self.video_config)
                full_res = self.picam2.sensor_resolution
                crop_x = int(full_res[0] * self.crop_x_ratio)
                crop_y = int(full_res[1] * self.crop_y_ratio)
                crop_w = int(full_res[0] * self.crop_w_ratio)
                crop_h = int(full_res[1] * self.crop_h_ratio)
                self.picam2.set_controls({"ScalerCrop": (crop_x, crop_y, crop_w, crop_h)})
                self.picam2.start()
                self.timer.start(30)  # 恢复视频流更新
            except Exception as e:
                self.status_label.setText(f'拍照失败: {str(e)}')
        else:
            self.status_label.setText('摄像头未启动，无法拍照')
    
    def load_image(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "打开图片",
                "/home/pi/Desktop/figures",
                "图片文件 (*.jpg *.jpeg *.png);;所有文件 (*.*)"
            )

            self.status_label.setText(f'文件对话框返回: {filename}')

            if filename:
                img = cv2.imread(filename)
                if img is not None:
                    self.last_image = img.copy()

                    if self.net is not None:
                        self.detect_button.setEnabled(True)
                    if self.is_camera_active:
                        self.toggle_camera()
                    self.display_image(img)
                    self.status_label.setText(f'已加载图片: {filename}')
                else:
                    self.status_label.setText(f'无法读取图片: {filename}')
            else:
                self.status_label.setText('未选择图片文件')
        except Exception as e:
            self.status_label.setText(f'加载图片时出错: {str(e)}')
        finally:
            self.result_table.hide()  # Hide table
            self.status_label.show()  # Show text area
    
    def check_reference_points(self, boxes):
        ref_status = [0] * len(self.ref_points)
        ref_points = self.ref_points
        for box in boxes:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            center = np.array([center_x, center_y])
            distances = np.linalg.norm(ref_points - center, axis=1)
            nearest_index = np.argmin(distances)
            ref_points[nearest_index] = ref_points[nearest_index] * 0
            ref_status[nearest_index] = 1

        # Generate table data
        table_data = []
        chunk = [ref_status[i:i + 3] for i in range(0, len(ref_status), 4)]
        for id in chunk:
            id = list(map(lambda status: '+' if status == 1 else '-', id))
            result = f"{id.count('+')} + , {id.count('-')} -"
            table_data.append(id + [result])

        # Update table
        self.result_table.setRowCount(len(table_data))
        for row_index, row_data in enumerate(table_data):
            for col_index, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data))
                item.setTextAlignment(Qt.AlignCenter)
                self.result_table.setItem(row_index, col_index, item)

        # Show table and hide text area
        self.result_table.show()
        self.status_label.hide()

    def resizeEvent(self, event):
        self.fixed_display_size = self.image_label.size()
        print(f'窗口大小变化，更新固定显示尺寸: {self.fixed_display_size}')
        self.display_image()
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        if self.is_camera_active:
            self.timer.stop()
            self.cap.release()
        event.accept()

if __name__ == "__main__":

    matplotlib.use('Agg')
    
    app = QApplication(sys.argv)
    window = YOLOv5LiteApp()
    window.showMaximized()
    sys.exit(app.exec_())
