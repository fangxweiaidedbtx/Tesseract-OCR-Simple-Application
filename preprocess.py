import cv2
import numpy as np


def preprocess(imgs):
    """
    预处理图片，包括灰度化，去噪和二值化
    :param imgs:  输入图片, 可以是列表或numpy数组
    :return:  预处理后的图片
    """
    def process_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 二值化
        return thresh

    if isinstance(imgs, list):
        imgs = [process_image(img) for img in imgs]
    elif isinstance(imgs, np.ndarray):
        imgs = [process_image(imgs)]

    return imgs










def main():
    img = cv2.imread("test6.jpg")
    img = preprocess(img)
    cv2.imshow("img", img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()