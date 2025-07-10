# vision_module.py (Corrected Version)

import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# --- 全局配置 (保持不变) ---
IMG_SIZE = 640
CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.5
CLASSES = ('wrench', 'hammer', 'file', 'tape_measure', 'multimeter',
           'pliers', 'screwdrivers', 'safety_goggles', 'feeler_gauge', 'vernier_caliper')

# --- 辅助函数 (保持不变) ---
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def postprocess(outputs, ratio, pad):
    predictions = np.squeeze(outputs[0])
    obj_conf_mask = predictions[:, 4] > CONF_THRESHOLD
    predictions = predictions[obj_conf_mask]
    if not predictions.shape[0]: return [], [], []
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1) * predictions[:, 4]
    score_mask = scores > CONF_THRESHOLD
    predictions = predictions[score_mask]
    class_ids = class_ids[score_mask]
    scores = scores[score_mask]
    if not predictions.shape[0]: return [], [], []
    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    x1 = cx - w / 2; y1 = cy - h / 2
    boxes = np.column_stack((x1, y1, w, h))
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0: return [], [], []
    indices = indices.flatten()
    final_boxes = boxes[indices]
    final_scores = scores[indices]
    final_class_ids = class_ids[indices]
    final_boxes[:, 0] = (final_boxes[:, 0] - pad[0]) / ratio
    final_boxes[:, 1] = (final_boxes[:, 1] - pad[1]) / ratio
    final_boxes[:, 2] /= ratio
    final_boxes[:, 3] /= ratio
    final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, None)
    final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, None)
    return final_boxes, final_scores, final_class_ids

def draw_results(image, boxes, scores, class_ids, CLASSES):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, w, h = box.astype(int)
        label = f"{CLASSES[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


class ObjectDetector:
    def __init__(self, model_path, camera_index):
        print("--- Initializing Vision Module (Model & Camera) ---")
        self.rknn_lite = RKNNLite(verbose=False)
        print(f'--> Loading RKNN model: {model_path}')
        if self.rknn_lite.load_rknn(model_path) != 0: exit(f"Failed to load RKNN model: {model_path}")
        print('--> Init runtime environment')
        if self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0: exit("Failed to init RKNN runtime")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            exit(-1)
        self.IMG_SIZE = IMG_SIZE
        self.CLASSES = CLASSES
        print("--- Vision Module Initialized Successfully ---")

    def search_for_object_live(self, target_label):
        """
        *** 【已修正的核心功能】 ***
        """
        print(f"\nLive searching for '{target_label}'... Press 'q' to cancel.")
        live_window_name = "Live Search - Looking for " + target_label
        
        found_counter = 0
        CONFIRMATION_THRESHOLD = 5  # 连续检测到5帧才算成功，防止抖动
        
        final_result_frame = None
        was_successful = False
        was_cancelled = False

        while True:
            # 1. 捕获帧
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame from camera.")
                break

            # 2. 推理
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_processed, ratio, pad = letterbox(img_rgb, new_shape=(self.IMG_SIZE, self.IMG_SIZE))
            outputs = self.rknn_lite.inference(inputs=[np.expand_dims(img_processed, axis=0)])
            
            # 3. 后处理和状态更新
            display_frame = frame.copy()
            target_this_frame = False
            
            if outputs:
                boxes, scores, class_ids = postprocess(outputs, ratio, pad)
                display_frame = draw_results(display_frame, boxes, scores, class_ids, self.CLASSES)
                if target_label in [self.CLASSES[cid] for cid in class_ids]:
                    target_this_frame = True
            
            if target_this_frame:
                found_counter += 1
            else:
                found_counter = 0 # 如果断了，就重置计数器

            # 4. 在实时画面上绘制状态信息
            if found_counter > 0:
                progress_text = f"Confirming: {found_counter}/{CONFIRMATION_THRESHOLD}"
                cv2.putText(display_frame, progress_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 5. 显示实时窗口 (这是每帧都必须做的)
            cv2.imshow(live_window_name, display_frame)

            # 6. 等待并处理按键 (这是让GUI刷新的关键!)
            key = cv2.waitKey(1) & 0xFF

            # 7. 在所有绘制和显示操作之后，再检查退出条件
            if found_counter >= CONFIRMATION_THRESHOLD:
                print(f"==> Target '{target_label}' CONFIRMED!")
                was_successful = True
                final_result_frame = display_frame # 保存这一帧作为最终结果
                break # 退出循环

            if key == ord('q'):
                print("Search cancelled by user.")
                was_cancelled = True
                final_result_frame = frame # 保存原始帧用于显示取消信息
                break # 退出循环

        # --- 循环结束后的清理和返回逻辑 ---
        cv2.destroyWindow(live_window_name)

        if was_successful:
            status_text = f"SUCCESS: Found {target_label}"
            cv2.putText(final_result_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return True, final_result_frame
        
        if was_cancelled:
            status_text = f"CANCELLED: Search for {target_label} was cancelled."
            cv2.putText(final_result_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return False, final_result_frame

        # 如果循环因其他原因退出 (如摄像头断开)
        return False, None


    def release(self):
        """释放所有资源"""
        print("--- Releasing vision resources... ---")
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.rknn_lite.release()
        cv2.destroyAllWindows()
        print("Vision resources released.")
