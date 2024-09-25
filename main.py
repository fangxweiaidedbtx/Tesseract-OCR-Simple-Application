from tesseract_functions import *
from preprocess import *

import pathlib
import cv2

config_default = r'-l chi_sim+eng '


def ocr(image_path):
    """
    识别图片中的文字
    :param img:  输入图片
    :return:  识别出的文字
    """
    if pathlib.Path(image_path).exists():
        img = cv2.imread(image_path)
        img = preprocess(img)
        data = pytesseract.image_to_data(img, config=config_default)
        text = ""
        print(data.splitlines())
        for  box in data.splitlines()[1:]:
            box = box.split('\t')
            if len(box) == 12:
                x, y, w, h = int(box[6]), int(box[7]), int(box[8]), int(box[9])
                cv2.rectangle(img, (x, y), (w + x, h + y), (0, 255, 0), 2)
                if box[11] != '' and box[11] != '-1':
                    text += box[11] + "\n"
        cv2.imshow("draw picture", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(text)
        return text
    else:
        print("Error: Image not found")


def read_text_from_image(image):
    """Reads text from an image file and outputs found text to text file"""
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OTSU Threshold
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)

    contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow("dilation", dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_copy = image.copy()
    text = ""
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cropped = image_copy[y: y + h, x: x + w]

        cv2.imshow("cropped", cropped)
        cv2.waitKey(0)

        text += pytesseract.image_to_string(cropped, config=config_default)
        print(text)
    return text


if __name__ == '__main__':
    image_paths = ["test6.jpg"]
    # image_paths = ["test.jpg", "test2.jpg", "test3.png", "test4.png","test5.png","test6.jpg"]
    for i in image_paths:
        ocr(i)
        # read_text_from_image(cv2.imread(i))
