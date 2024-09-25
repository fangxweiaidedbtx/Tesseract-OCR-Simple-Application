import cv2
import numpy as np
import pytesseract
import pathlib
from math import fabs


def preprocess(imgs):
    """
    预处理图片，包括灰度化，去噪和二值化
    :param imgs:  输入图片, 可以是列表或numpy数组
    :return:  预处理后的图片
    """

    def process_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 二值化
        return gray_image

    if isinstance(imgs, list):
        # 如果imgs是列表类型，则对列表中的每个图像进行处理
        imgs = [process_image(img) for img in imgs]
        return imgs

    elif isinstance(imgs, np.ndarray):
        # 如果imgs是numpy数组类型(图像)，则对整个数组进行处理
        imgs = process_image(imgs)
        return imgs


def ocr(image_path, config_default=r'-l chi_sim+eng ',show_image=False):
    """
    识别图片中的文字
    :param image_path:  输入图片
    :param config_default:  识别参数
    :return:  识别出的文字
    """
    # 判断image_path是否存在
    if not pathlib.Path(image_path).exists():
        print("image_path not exist")
        return None
    # 读取图片
    img = cv2.imread(image_path)
    # 预处理图片
    img = preprocess(img)
    # 使用pytesseract识别图片中的文字，并返回识别结果
    data = pytesseract.image_to_data(img, config=config_default)
    # 创建一个空列表，用于存储识别到的文字
    texts = []
    # 遍历识别结果
    for box in data.splitlines()[1:]:
        # 将识别结果按制表符分割
        box = box.split('\t')
        # 判断识别结果是否包含12个元素
        if len(box) == 12:
            # 提取识别结果的坐标
            x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
            # 计算文字的起始点和结束点
            start_point = (x, y)
            end_point = (x + w, y + h)
            # 在图片上绘制矩形框
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
            # 判断识别结果是否包含文字
            if box[11] != '' and box[11] != '-1':
                # 将识别结果添加到列表中
                text = {"start_point": start_point, "end_point": end_point, "text": box[11] + "\t"}
                texts.append(text)
    # 如果show_image为True，则显示图片
    if show_image:
        cv2.imshow("draw picture", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # 返回识别到的文字
    return texts


def print_text(texts: dict, mathod="raw"):
    """
    打印识别出的文字
    :param texts:  识别出的字典，形状为{"start_point": (x, y), "end_point": (x, y), "text": str}
    :param mathod:  打印方式，"raw"表示原始输出，"row"表示按行内打印,
    :return:  None
    """
    result = ""
    # 如果mathod等于"raw"，则将texts中的每个text的"text"属性连接起来，并在每个text之间添加换行符
    if mathod == "raw":
        for text in texts:
            result += text["text"] + "\n"
        print(result)

    # 如果mathod等于"row"，则将texts中的每个text的"text"属性连接起来，如果当前text的"start_point"的y坐标与下一个text的"start_point"的y坐标之差小于8，则不添加换行符
    elif mathod == "row":
        count = 0
        for text in texts:
            if count < len(texts) - 1 and fabs(text["start_point"][1] - texts[count + 1]["start_point"][1]) < 8:
                result += text["text"]
            else:
                result += text["text"] + "\n"
            count += 1
        print(result.strip())  # 输出时去除多余的换行符

def write_file(texts: dict, filename="output.txt",mathod="raw"):
    """
    将识别出的文字写入文件
    :param texts:  识别出的字典，形状为{"start_point": (x, y), "end_point": (x, y), "text": str}
    :param filename:  文件名，默认为"output.txt"
    :param mathod:  打印方式，"raw"表示原始输出，"row"表示按行内打印,
    :return:  None
    """
    result = ""
    # 如果mathod等于"raw"，则将texts中的每个text的"text"属性连接起来，并在每个text之间添加换行符
    if mathod == "raw":
        for text in texts:
            result += text["text"] + "\n"
        if pathlib.Path(filename).exists():
            with open(filename, "a", encoding="utf-8") as f:
                f.write(result)
        else:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)
    # 如果mathod等于"row"，则将texts中的每个text的"text"属性连接起来，并在每个text之间添加空格
    elif mathod == "row":
        count = 0
        for text in texts:
            if count < len(texts) - 1 and fabs(text["start_point"][1] - texts[count + 1]["start_point"][1]) < 8:
                result += text["text"]
            else:
                result += text["text"] + "\n"
            count += 1
        if pathlib.Path(filename).exists():
            with open(filename, "a", encoding="utf-8") as f:
                f.write(result)
        else:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)

def main():
    img = cv2.imread("test5.jpg")
    img = preprocess(img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
