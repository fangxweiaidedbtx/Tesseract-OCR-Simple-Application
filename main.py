from preprocess import *

if __name__ == '__main__':
    image_paths = ["test7.png","test.png"]
    # 遍历图片路径列表
    for i in image_paths:
        # 调用ocr函数，对图片进行识别，show_image参数设置为False，不显示识别结果
        texts = ocr(i,show_image=True)
        # 调用print_text函数，打印识别结果，mathod参数设置为'row'，按行打印
        print_text(texts,mathod='row')
        write_file(texts, filename=f"{i}.txt",mathod="row")