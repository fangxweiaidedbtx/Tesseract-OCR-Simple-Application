from preprocess import *
#效率测试
import time
if __name__ == '__main__':
    image_paths = ["test7.png","test.png"]
    #效率测试
    start_time = time.time()
    for j in range(1,11):
        for i in image_paths:
            texts = ocr(i,show_image=False)#为了测试识别效率，不显示图片信息
            # print_text(texts,mathod='row')
            write_file(texts, filename=f"{i}.txt",mathod="row")
    end_time = time.time()
    print(f"识别{len(image_paths)*10}张图片，用时{end_time-start_time}秒")