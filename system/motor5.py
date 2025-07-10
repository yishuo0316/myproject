import gpiod
import time
import threading

# ===== 引脚配置 =====
# 电机A方向控制引脚 (使用 gpiochip3)
AIN1_PIN = 13   # GPIO3_B5 -> 线号13 (方向控制1)
AIN2_PIN = 8    # GPIO3_B0 -> 线号8 (方向控制2)
PWMA_PIN = 12   # GPIO3_B4 -> 线号12 (PWM调速)

# 电机B方向控制引脚 (使用 gpiochip3)
BIN1_PIN = 2    # GPIO3_A2
BIN2_PIN = 3    # GPIO3_A3 
PWMB_PIN = 4    # GPIO3_A4

# 循迹红外传感器引脚 (使用 gpiochip3)
TRACK_LEFT1_PIN = 0   # 左边第一个循迹传感器
TRACK_LEFT2_PIN = 5   # 左边第二个循迹传感器
TRACK_RIGHT1_PIN = 6  # 右边第一个循迹传感器
TRACK_RIGHT2_PIN = 7  # 右边第二个循迹传感器

GPIO_CHIP = 'gpiochip3'

# ===== PWM 参数 =====
PWM_FREQUENCY = 500  # 降低频率以减少电机噪声

class PWMController:
    """PWM 控制器"""
    def __init__(self, chip, pin, frequency=500):
        self.chip_name = chip
        self.pin = pin
        self.frequency = frequency
        self.duty_cycle = 0
        self.running = False
        self.thread = None
        
        # 初始化 GPIO
        self.chip = gpiod.Chip(chip)
        self.line = self.chip.get_line(pin)
        self.line.request(consumer="pwm", type=gpiod.LINE_REQ_DIR_OUT)
        
    def start(self):
        """启动 PWM 输出"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._pwm_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """停止 PWM 输出"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.line.set_value(0)
    
    def set_duty_cycle(self, duty_cycle):
        """设置 PWM 占空比 (0-100)"""
        self.duty_cycle = max(0, min(100, duty_cycle))
    
    def _pwm_loop(self):
        """PWM 输出循环"""
        period = 1.0 / self.frequency
        try:
            while self.running:
                if self.duty_cycle > 0:
                    on_time = period * (self.duty_cycle / 100.0)
                    self.line.set_value(1)
                    time.sleep(on_time)
                
                if self.duty_cycle < 100:
                    off_time = period * ((100 - self.duty_cycle) / 100.0)
                    self.line.set_value(0)
                    time.sleep(off_time)
        except Exception as e:
            print(f"PWM error: {e}")
        finally:
            self.line.set_value(0)

class MotorController:
    def __init__(self):
        # 初始化方向控制引脚
        self.chip = gpiod.Chip(GPIO_CHIP)
        
        # 电机A引脚
        self.ain1 = self.chip.get_line(AIN1_PIN)
        self.ain2 = self.chip.get_line(AIN2_PIN)
        
        # 电机B引脚
        self.bin1 = self.chip.get_line(BIN1_PIN)
        self.bin2 = self.chip.get_line(BIN2_PIN)
        
        # 循迹传感器引脚
        self.track_left1 = self.chip.get_line(TRACK_LEFT1_PIN)
        self.track_left2 = self.chip.get_line(TRACK_LEFT2_PIN)
        self.track_right1 = self.chip.get_line(TRACK_RIGHT1_PIN)
        self.track_right2 = self.chip.get_line(TRACK_RIGHT2_PIN)
        
        # 请求控制权
        self.ain1.request(consumer="motorA", type=gpiod.LINE_REQ_DIR_OUT)
        self.ain2.request(consumer="motorA", type=gpiod.LINE_REQ_DIR_OUT)
        self.bin1.request(consumer="motorB", type=gpiod.LINE_REQ_DIR_OUT)
        self.bin2.request(consumer="motorB", type=gpiod.LINE_REQ_DIR_OUT)
        
        # 初始化循迹传感器为输入
        self.track_left1.request(consumer="track_left1", type=gpiod.LINE_REQ_DIR_IN)
        self.track_left2.request(consumer="track_left2", type=gpiod.LINE_REQ_DIR_IN)
        self.track_right1.request(consumer="track_right1", type=gpiod.LINE_REQ_DIR_IN)
        self.track_right2.request(consumer="track_right2", type=gpiod.LINE_REQ_DIR_IN)
        
        # 初始化 PWM 控制器
        self.pwm_a = PWMController(GPIO_CHIP, PWMA_PIN, PWM_FREQUENCY)
        self.pwm_b = PWMController(GPIO_CHIP, PWMB_PIN, PWM_FREQUENCY)
        self.pwm_a.start()
        self.pwm_b.start()
        
        # 当前速度
        self.speed_a = 30  # 默认速度降低为30%
        self.speed_b = 30
    
    def set_speed(self, motor, speed):
        """设置电机速度 (0-100)"""
        speed = max(0, min(100, speed))
        if motor.upper() == 'A':
            self.speed_a = speed
            self.pwm_a.set_duty_cycle(speed)
        elif motor.upper() == 'B':
            self.speed_b = speed
            self.pwm_b.set_duty_cycle(speed)
    
    def forward(self, motor, speed=None):
        """前进"""
        if speed is not None:
            if motor.upper() == 'ALL':
                self.set_speed('A', speed)
                self.set_speed('B', speed)
            else:
                self.set_speed(motor, speed)
        
        if motor.upper() == 'A' or motor.upper() == 'ALL':
            self.ain1.set_value(1)
            self.ain2.set_value(0)
        
        if motor.upper() == 'B' or motor.upper() == 'ALL':
            self.bin1.set_value(1)
            self.bin2.set_value(0)
    
    def backward(self, motor, speed=None):
        """后退"""
        if speed is not None:
            if motor.upper() == 'ALL':
                self.set_speed('A', speed)
                self.set_speed('B', speed)
            else:
                self.set_speed(motor, speed)
        
        if motor.upper() == 'A' or motor.upper() == 'ALL':
            self.ain1.set_value(0)
            self.ain2.set_value(1)
        
        if motor.upper() == 'B' or motor.upper() == 'ALL':
            self.bin1.set_value(0)
            self.bin2.set_value(1)
    
    def stop(self, motor='ALL'):
        """停止"""
        if motor.upper() == 'A' or motor.upper() == 'ALL':
            self.ain1.set_value(0)
            self.ain2.set_value(0)
            if motor.upper() == 'ALL':
                self.pwm_a.set_duty_cycle(0)
        
        if motor.upper() == 'B' or motor.upper() == 'ALL':
            self.bin1.set_value(0)
            self.bin2.set_value(0)
            if motor.upper() == 'ALL':
                self.pwm_b.set_duty_cycle(0)
    
    def read_track_sensors(self):
        """读取循迹传感器状态"""
        left1 = self.track_left1.get_value() == 0  # 检测到黑线为True
        left2 = self.track_left2.get_value() == 0
        right1 = self.track_right1.get_value() == 0
        right2 = self.track_right2.get_value() == 0
        
        return left1, left2, right1, right2
    
    def tracking_move(self):
        """根据循迹传感器状态控制小车移动"""
        left1, left2, right1, right2 = self.read_track_sensors()
        
        # 关键改进：所有传感器均未检测到黑线时停止
        if not (left1 or left2 or right1 or right2):
            self.stop()
            return
        
        # 低速循迹逻辑（速度参数已优化）
        if (left1 or left2) and right2:
            self.spin_right(15, 15)
            time.sleep(0.05)
        elif left1 and (right1 or right2):
            self.spin_left(15, 15)
            time.sleep(0.05)
        elif left1:
            self.spin_left(12, 12)
        elif right2:
            self.spin_right(12, 12)
        elif left2 and not right1:
            self.left(0, 15)
        elif not left2 and right1:
            self.right(15, 0)
        elif left2 and right1:
            self.run(15, 15)
    
    def run(self, left_speed, right_speed):
        """小车前进"""
        self.set_speed('A', left_speed)
        self.set_speed('B', right_speed)
        self.forward('ALL')
    
    def left(self, left_speed, right_speed):
        """小车左转（右轮前进，左轮停止）"""
        self.set_speed('A', left_speed)
        self.set_speed('B', right_speed)
        self.stop('A')
        self.forward('B')
    
    def right(self, left_speed, right_speed):
        """小车右转（左轮前进，右轮停止）"""
        self.set_speed('A', left_speed)
        self.set_speed('B', right_speed)
        self.forward('A')
        self.stop('B')
    
    def spin_left(self, left_speed, right_speed):
        """小车原地左转（右轮前进，左轮后退）"""
        self.set_speed('A', left_speed)
        self.set_speed('B', right_speed)
        self.backward('A')
        self.forward('B')
    
    def spin_right(self, left_speed, right_speed):
        """小车原地右转（左轮前进，右轮后退）"""
        self.set_speed('A', left_speed)
        self.set_speed('B', right_speed)
        self.forward('A')
        self.backward('B')
    
    def cleanup(self):
        """清理资源"""
        self.stop('ALL')
        self.pwm_a.stop()
        self.pwm_b.stop()
        self.ain1.release()
        self.ain2.release()
        self.bin1.release()
        self.bin2.release()
        self.track_left1.release()
        self.track_left2.release()
        self.track_right1.release()
        self.track_right2.release()
        self.chip.close()

if __name__ == "__main__":
    motor = MotorController()
    try:
        print("循迹小车启动（检测不到黑线自动停止）...")
        time.sleep(2)
        
        while True:
            motor.tracking_move()
            time.sleep(0.02)  # 更快的控制周期
    
    except KeyboardInterrupt:
        print("\n手动停止")
    finally:
        motor.cleanup()