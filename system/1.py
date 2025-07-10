import subprocess
import time
import cv2
import os
import threading
from vision_module import ObjectDetector
from motor5 import MotorController

# --- 配置 ---
SOUND_APP_PATH = './soundapp'
SERIAL_PORT = '/dev/ttyS9'
MODEL_PATH = './yolov5.rknn'
CAMERA_INDEX = 21
DISPLAY_DURATION_MS = 1000

KEYWORD_MAP = {
    "扳手": "wrench",
    "锤子": "hammer",
    "锉刀": "file",
    "卷尺": "tape_measure",
    "万用表": "multimeter",
    "钳子": "pliers",
    "螺丝刀": "screwdrivers",
    "护目镜": "safety_goggles",
    "塞尺": "feeler_gauge",
    "游标卡尺": "vernier_caliper"
}


def run_tracking_in_background(motor, stop_event):
    """
    此函数在后台线程运行，负责小车循迹。
    它会持续运行，直到主线程通过 stop_event 通知它停止。
    """
    print("[线程] 循迹功能已在后台启动。")
    try:
        while not stop_event.is_set():
            motor.tracking_move()
            time.sleep(0.02)  # 适当延时，防止CPU满载
    finally:
        motor.stop()
        print("[线程] 循迹功能已停止。")


def main():
    # 1. 初始化所有模块
    detector = ObjectDetector(model_path=MODEL_PATH, camera_index=CAMERA_INDEX)
    motor = MotorController()

    # 2. 在后台启动语音识别子进程
    print(f"启动语音识别程序: {SOUND_APP_PATH}")
    try:
        voice_process = subprocess.Popen(
            [SOUND_APP_PATH, SERIAL_PORT],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1, text=True, encoding='utf-8', errors='ignore'
        )
    except Exception as e:
        print(f"错误：无法启动语音程序: {e}")
        return

    print("=" * 30)
    print("系统准备就绪，等待语音指令...")
    print("=" * 30)

    try:
        # 3. 循环监听语音指令
        for line in voice_process.stdout:
            line = line.strip()
            if not line:
                continue

            print(f"[语音识别]: {line}")

            target_keyword_en = None
            for keyword_cn, keyword_en in KEYWORD_MAP.items():
                if keyword_cn in line and "识别成功" in line:
                    print(f"\n>>> 收到指令: 开始寻找 '{keyword_cn}' ({keyword_en})")
                    target_keyword_en = keyword_en
                    break

            if target_keyword_en:
                ### 4. 收到有效指令，开始并发执行任务 ###

                # a. 创建一个事件，用于在任务完成后通知循迹线程停止
                stop_tracking_event = threading.Event()

                # b. 创建并启动循迹线程 (后台任务)
                tracking_thread = threading.Thread(
                    target=run_tracking_in_background,
                    args=(motor, stop_tracking_event)
                )
                tracking_thread.start()

                # c. 在主线程中执行视觉搜索 (前台任务，会阻塞直到完成)
                found, result_frame = detector.search_for_object_live(target_keyword_en)

                # d. 视觉搜索结束，立即停止循迹线程
                print(">>> 视觉搜索结束，正在停止小车...")
                stop_tracking_event.set()
                tracking_thread.join()  # 等待循迹线程安全退出

                # e. 处理并显示最终结果
                if found and result_frame is not None:
                    print(f"✔ 任务成功! 已找到 {target_keyword_en}.")
                    cv2.imshow("Target Found!", result_frame)
                    cv2.waitKey(DISPLAY_DURATION_MS)
                    cv2.destroyWindow("Target Found!")
                else:
                    print(f"✖ 任务失败. 未能确认找到 {target_keyword_en}。")

                print("\n" + "=" * 30)
                print("系统准备就绪，等待下一条语音指令...")
                print("=" * 30)

    except KeyboardInterrupt:
        print("\n程序被用户终止 (Ctrl+C)")
    finally:
        # 5. 清理所有资源
        print("正在清理资源并关闭系统...")
        if 'voice_process' in locals() and voice_process.poll() is None:
            voice_process.terminate()
            voice_process.wait(timeout=2)

        # 确保所有可能运行的线程都被通知停止
        if 'stop_tracking_event' in locals() and not stop_tracking_event.is_set():
            stop_tracking_event.set()
        if 'tracking_thread' in locals() and tracking_thread.is_alive():
            tracking_thread.join()

        detector.release()
        motor.cleanup()
        cv2.destroyAllWindows()
        print("系统已安全关闭。")


if __name__ == '__main__':
    # 确保显示环境可用
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'
        print("警告: DISPLAY 环境变量未设置, 已自动设为 ':0'")

    if not os.path.exists(SOUND_APP_PATH) or not os.access(SOUND_APP_PATH, os.X_OK):
        print(f"错误: 语音程序 '{SOUND_APP_PATH}' 不存在或没有执行权限!")
    else:
        main()