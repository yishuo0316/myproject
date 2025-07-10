import sys
from rknn.api import RKNN

DATASET_PATH = '/home/elf/Desktop/yolov8_rknn/dataset.txt'

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path platform [output_rknn_path]".format(sys.argv[0]))
        exit(1)
    model_path = sys.argv[1]
    platform = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'yolov5.rknn'
    return model_path, platform, output_path

if __name__ == '__main__':
    model_path, platform, output_path = parse_arg()
    rknn = RKNN(verbose=True)
    print('--> Config model')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=platform,
        #quantized_dtype='w8a8',
        output_optimize=True,
    )
    print('done')
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0: rknn.release(); exit(ret)
    print('done')
    print('--> Building model...')
    ret = rknn.build(do_quantization=False)
    if ret != 0: rknn.release(); exit(ret)
    print('done')
    print('--> Exporting rknn model...')
    ret = rknn.export_rknn(output_path)
    if ret != 0: rknn.release(); exit(ret)
    print(f'--> RKNN model saved to {output_path}')
    print('done')
    rknn.release()
