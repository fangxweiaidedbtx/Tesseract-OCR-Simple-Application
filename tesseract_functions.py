import cv2
import pytesseract
def draw_by_tesseraact(img,config='--psm 6'):
    """
    使用tesseract进行文字检测并画框
    :param image:
    :return: 返回经过tesseract检测的图像
    """
    boxes=pytesseract.image_to_boxes(img,config=config)
    if len(img.shape)==2:
        h, w= img.shape
    if len(img.shape)==3:
        h,w,c = img.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
    return img