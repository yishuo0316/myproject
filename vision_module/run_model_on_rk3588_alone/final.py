import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite
import argparse

# --- 全局配置 ---
IMG_SIZE = 640  # 模型输入尺寸
CONF_THRESHOLD = 0.45  # 目标置信度阈值
NMS_THRESHOLD = 0.5  # NMS阈值
CLASSES = ('wrench', 'hammer', 'file', 'tape_measure', 'multimeter',
           'pliers', 'screwdrivers', 'safety_goggles', 'feeler_gauge', 'vernier_caliper')


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    YOLOv5的letterbox预处理函数。
    将图像调整大小并填充以匹配模型输入尺寸，同时保持纵横比。
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def postprocess(outputs, ratio, pad):
    """
    对模型输出进行后处理，解码边界框。
    """
    # 输出形状为 [1, 25200, 15]
    predictions = np.squeeze(outputs[0])

    # 1. 过滤掉低置信度的框
    obj_conf_mask = predictions[:, 4] > CONF_THRESHOLD
    predictions = predictions[obj_conf_mask]

    if not predictions.shape[0]:
        return [], [], []

    # 2. 获取类别和分数
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1) * predictions[:, 4]  # 最终分数 = 类别分数 * 目标置信度

    # 再次过滤，确保最终分数也满足阈值
    score_mask = scores > CONF_THRESHOLD
    predictions = predictions[score_mask]
    class_ids = class_ids[score_mask]
    scores = scores[score_mask]

    if not predictions.shape[0]:
        return [], [], []

    # 3. 将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
    box_coords = predictions[:, :4]
    cx, cy, w, h = box_coords[:, 0], box_coords[:, 1], box_coords[:, 2], box_coords[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    boxes = np.column_stack((x1, y1, w, h))  # NMS函数需要x,y,w,h格式

    # 4. 应用NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) == 0:
        return [], [], []

    final_boxes = boxes[indices]
    final_scores = scores[indices]
    final_class_ids = class_ids[indices]

    # 5. 将坐标从640x640空间映射回原始图像空间
    final_boxes[:, 0] = (final_boxes[:, 0] - pad[0]) / ratio  # x
    final_boxes[:, 1] = (final_boxes[:, 1] - pad[1]) / ratio  # y
    final_boxes[:, 2] = final_boxes[:, 2] / ratio  # w
    final_boxes[:, 3] = final_boxes[:, 3] / ratio  # h

    # 确保坐标在图像范围内
    final_boxes[:, 0] = np.clip(final_boxes[:, 0], 0, None)
    final_boxes[:, 1] = np.clip(final_boxes[:, 1], 0, None)

    return final_boxes, final_scores, final_class_ids


def draw_results(image, boxes, scores, class_ids):
    """在图像上绘制检测结果"""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, w, h = box.astype(int)
        x1, y1, x2, y2 = x, y, x + w, y + h
        label = f"{CLASSES[class_id]}: {score:.2f}"

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制标签背景
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)

        # 绘制标签文本
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return image


def main(args):
    # 1. 模型初始化
    rknn_lite = RKNNLite(verbose=False)

    print(f'--> Loading RKNN model: {args.model_path}')
    ret = rknn_lite.load_rknn(args.model_path)
    if ret != 0:
        print(f'Load RKNN model failed! Ret = {ret}')
        exit(ret)

    print('--> Init runtime environment')
    # 对于RK3588, core_mask可以设置为RKNNLite.NPU_CORE_0_1_2以使用所有三个NPU核心
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print(f'Init runtime environment failed! Ret = {ret}')
        exit(ret)
    print('done')

    # 2. 读取和预处理图像
    print(f'--> Reading image: {args.image_path}')
    orig_img = cv2.imread(args.image_path)
    if orig_img is None:
        print(f"Failed to read image: {args.image_path}")
        return

    # Letterbox + BGR to RGB
    # 注意：我们传入的是 uint8 图像，因为归一化已在rknn模型中配置
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_processed, ratio, pad = letterbox(img_rgb, new_shape=(IMG_SIZE, IMG_SIZE))

    img_processed = np.expand_dims(img_processed, axis=0)
    # img_processed = img_processed.transpose(0, 3, 1, 2)
    # 3. 模型推理
    print('--> Inference')
    # rknnlite.inference的输入需要是一个列表
    # 输入数据类型应为uint8，因为rknn.config中已指定了std_values
    outputs = rknn_lite.inference(inputs=[img_processed])
    if outputs is None:
        print('Inference failed!')
        rknn_lite.release()
        exit(-1)
    print('done')

    # **************************
    # *** 在这里添加诊断代码 ***
    # **************************
    print("\n" + "=" * 20 + " RAW OUTPUT DIAGNOSIS " + "=" * 20)
    raw_output = outputs[0]
    print(f"Raw output shape: {raw_output.shape}")
    print(f"Raw output dtype: {raw_output.dtype}")
    print(
        f"Raw output stats: Min={np.min(raw_output):.4f}, Max={np.max(raw_output):.4f}, Mean={np.mean(raw_output):.4f}")

    # 打印第4个通道（目标置信度）的统计信息，这是最关键的
    objectness_scores = raw_output[0, :, 4]
    print(
        f"Objectness scores stats: Min={np.min(objectness_scores):.4f}, Max={np.max(objectness_scores):.4f}, Mean={np.mean(objectness_scores):.4f}")
    print("=" * 62 + "\n")
    # **************************
    # *** 诊断代码结束 ***
    # **************************

    # 4. 后处理
    print('--> Post-processing')
    boxes, scores, class_ids = postprocess(outputs, ratio, pad)
    print(f"Found {len(boxes)} objects.")

    # 5. 绘制结果并保存
    if len(boxes) > 0:
        result_img = draw_results(orig_img, boxes, scores, class_ids)

        # 创建输出目录
        output_dir = "inference_results"
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果
        base_name = os.path.basename(args.image_path)
        save_path = os.path.join(output_dir, f"result_{base_name}")
        cv2.imwrite(save_path, result_img)
        print(f"Result saved to {save_path}")

    # 6. 释放模型
    rknn_lite.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 RKNN Inference on RK3588")
    parser.add_argument('--model_path', type=str, default='./yolov5.rknn', help='Path to the rknn model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    main(args)
