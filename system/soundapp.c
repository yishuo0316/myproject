#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

// 初始化串口 (无变动)
int init_uart(int fd)
{
	struct termios newtio, oldtio;
	if (tcgetattr(fd, &oldtio) != 0) {
		perror("tcgetattr");
		return -1;
	}
	bzero(&newtio, sizeof(newtio));
	newtio.c_cflag |= CLOCAL | CREAD;
	newtio.c_cflag &= ~CSIZE;
	newtio.c_cflag |= CS8;
	newtio.c_cflag &= ~PARENB;
	cfsetispeed(&newtio, B9600);
	cfsetospeed(&newtio, B9600);
	newtio.c_cflag &= ~CSTOPB;
	newtio.c_cc[VTIME] = 0;
	newtio.c_cc[VMIN] = 0;
	tcflush(fd, TCIFLUSH);
	if ((tcsetattr(fd, TCSANOW, &newtio)) != 0) {
		perror("com set error");
		return -1;
	}
	printf("✅ 串口设置完成。\n");
	return 0;
}

// 读取串口数据 (无变动)
int uart_read_frame(int fd, unsigned char *p_receive_buff, const int count, int timeout_data)
{
	int nread = 0;
	fd_set rd;
	int retval = 0;
	struct timeval timeout = {0, timeout_data * 1000};
	FD_ZERO(&rd);
	FD_SET(fd, &rd);
	memset(p_receive_buff, 0x0, count);
	retval = select(fd + 1, &rd, NULL, NULL, &timeout);
	if (retval > 0)
		nread = read(fd, p_receive_buff, count - 1);
	else if (retval == -1)
		perror("select error");
	return nread;
}

// 打印十六进制 (无变动)
void print_hex(const char* prefix, const unsigned char* data, int len) {
    if (len <= 0) return;
    printf("%s (%d bytes): ", prefix, len);
    for (int i = 0; i < len; i++) {
        printf("%02X ", data[i]);
    }
    printf("\n");
}


// --- 修正后的 GBK 编码常量 ---

// GBK-encoded byte sequences for our keywords
const unsigned char Gbk_Wrench[]      = {0xB0, 0xE2, 0xCA, 0xD6}; // 扳手
const unsigned char Gbk_Hammer[]      = {0xB4, 0xB8, 0xD7, 0xD3}; // 锤子
const unsigned char Gbk_File[]        = {0xC9, 0xEC, 0xB5, 0xD6}; // 锉刀 (已修正)
const unsigned char Gbk_Tape[]        = {0xBE, 0xED, 0xB3, 0xDF}; // 卷尺
const unsigned char Gbk_Multimeter[]  = {0xCD, 0xF2, 0xD3, 0xC3, 0xB1, 0xED}; // 万用表
const unsigned char Gbk_Pliers[]      = {0xC7, 0xCF, 0xD7, 0xD3}; // 钳子 (已修正)
const unsigned char Gbk_Screwdriver[] = {0xC2, 0xE5, 0xCB, 0xF9, 0xB5, 0xD6}; // 螺丝刀 (已修正)
const unsigned char Gbk_Goggles[]     = {0xBB, 0xA4, 0xC4, 0xBF, 0xBE, 0xB5}; // 护目镜
const unsigned char Gbk_FeelerGauge[] = {0xC8, 0xFB, 0xB3, 0xDF}; // 塞尺
const unsigned char Gbk_Caliper[]     = {0xD3, 0xCE, 0xB1, 0xEA, 0xBF, 0xA8, 0xB3, 0xDF}; // 游标卡尺

// 辅助函数：在缓冲区中查找一个字节序列 (无变动)
// ... (后面的代码保持不变) ...
const unsigned char* find_bytes(const unsigned char* haystack, int haystack_len, const unsigned char* needle, int needle_len) {
    if (needle_len == 0) return haystack;
    if (haystack_len < needle_len) return NULL;
    for (int i = 0; i <= haystack_len - needle_len; ++i) {
        if (memcmp(haystack + i, needle, needle_len) == 0) {
            return haystack + i;
        }
    }
    return NULL;
}

// 主函数 (无变动)
int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);

	if (argc < 2) {
		fprintf(stderr, "用法: %s <串口设备路径，例如 /dev/ttyS9>\n", argv[0]);
		return -1;
	}

	int fd = open(argv[1], O_RDWR | O_NOCTTY | O_NDELAY);
	if (fd < 0) {
		perror("❌ 打开串口失败");
		return -1;
	}

	if (init_uart(fd) != 0) {
		close(fd);
		return -1;
	}

	unsigned char buf[64];
	int nread;

	printf("🎤 正在监听语音模块...\n");

	while (1)
	{
		nread = uart_read_frame(fd, buf, sizeof(buf), 100);

		if (nread > 0) {
			print_hex("[调试] 串口原始数据", buf, nread);

			if (find_bytes(buf, nread, Gbk_Wrench, sizeof(Gbk_Wrench)))             { printf("扳手 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_Screwdriver, sizeof(Gbk_Screwdriver))) { printf("螺丝刀 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_Caliper, sizeof(Gbk_Caliper)))         { printf("游标卡尺 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_Pliers, sizeof(Gbk_Pliers)))           { printf("钳子 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_Hammer, sizeof(Gbk_Hammer)))           { printf("锤子 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_Tape, sizeof(Gbk_Tape)))               { printf("卷尺 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_Multimeter, sizeof(Gbk_Multimeter)))   { printf("万用表 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_File, sizeof(Gbk_File)))               { printf("锉刀 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_FeelerGauge, sizeof(Gbk_FeelerGauge))) { printf("塞尺 识别成功\n"); }
			else if (find_bytes(buf, nread, Gbk_Goggles, sizeof(Gbk_Goggles)))         { printf("护目镜 识别成功\n"); }
		}

		usleep(100000);
	}

	close(fd);
	return 0;
}