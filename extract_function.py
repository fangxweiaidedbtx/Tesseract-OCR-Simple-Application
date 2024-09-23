import cv2
import numpy as np
import pytesseract


def load_model(model_path):
    return cv2.dnn.readNet(model_path)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return None, None
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (123.68, 116.779, 103.939), swapRB=True, crop=False)
    return image, blob


def detect_by_east(net, image_blob):
    """
    检测文本区域
    :param net: 模型
    :param image_blob: 图像blob
    :return:
    """
    net.setInput(image_blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    return scores, geometry



def extract_areas(scores, geometry, threshold=0.5):
    """提取具有高置信度的区域。

    参数：
    scores : ndarray
        模型输出的得分矩阵。
    geometry : ndarray
        每个区域的几何信息。
    threshold : float
        置信度阈值，低于该值的得分将被忽略。

    返回：
    rects : list
        符合条件的矩形区域的坐标列表。
    confidences : list
        符合条件的区域的置信度值列表。
    """
    rects = []
    confidences = []
    (numRows, numCols) = scores.shape[2:4]
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(numCols):
            score = scoresData[x]
            if score < threshold:
                continue
            angle = anglesData[x]
            offsetX, offsetY = x * 4.0, y * 4.0
            angle_rad = angle * np.pi / 180.0
            cos, sin = np.cos(angle_rad), np.sin(angle_rad)
            h, w = x0[x] + x2[x], x1[x] + x3[x]
            endX, endY = int(offsetX + (cos * x1[x]) + (sin * x2[x])), int(offsetY - (sin * x1[x]) + (cos * x2[x]))
            startX, startY = int(endX - w), int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))
    return rects, confidences


def draw_rectangles(image, rects):
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 8)
    return image


def filter_duplicates(rects, confidences, threshold=0.9):
    """
    过滤重复的矩形框，根据置信度阈值和重叠情况进行去重
    :param rects:  矩形框列表
    :param confidences: 对应每个矩形框的置信度列表
    :param threshold: 置信度的阈值，默认为0.9
    :return:  过滤后的矩形框列表
    """
    filtered_rects = []  # 存储过滤后的矩形框

    for i in range(len(rects)):
        current_rect = rects[i]  # 当前矩形框
        current_confidence = confidences[i]  # 当前矩形框的置信度

        # 检查当前置信度是否大于阈值
        if current_confidence >= threshold:
            merged_rect = current_rect  # 初始化合并矩形框为当前框
            add_rect = True  # 标记是否添加当前矩形框

            # 遍历过滤后的矩形框，检查是否重叠
            for filtered_rect in filtered_rects:
                if is_overlap(merged_rect, filtered_rect):  # 判断矩形框是否重叠
                    merged_rect = merge_rects(merged_rect, filtered_rect)  # 合并矩形框
                    filtered_rects.remove(filtered_rect)  # 从过滤列表中删除被合并的矩形框
                    filtered_rects.append(merged_rect)  # 添加合并后的矩形框
                    add_rect = False  # 标记为不添加当前矩形框

            # 根据是否合并决定是否将框添加到结果中
            if add_rect:
                filtered_rects.append(merged_rect)  # 添加当前矩形框


    print("Filtered rectangles:", len(filtered_rects))  # 打印过滤后的矩形框数量
    print(filtered_rects)  # 打印过滤后的矩形框
    return filtered_rects  # 返回过滤后的矩形框列表

def merge_rects(rect1, rect2):
    x1_start, y1_start, x1_end, y1_end = rect1
    x2_start, y2_start, x2_end, y2_end = rect2

    merged_x1 = min(x1_start, x2_start)
    merged_y1 = min(y1_start, y2_start)
    merged_x2 = max(x1_end, x2_end)
    merged_y2 = max(y1_end, y2_end)

    return (merged_x1, merged_y1, merged_x2, merged_y2)

def is_overlap(rect1, rect2):
    """
    判断两个矩形框是否重叠
    :return:  True or False
    """
    x1_start, y1_start, x1_end, y1_end = rect1
    x2_start, y2_start, x2_end, y2_end = rect2

    inter_rect_x1 = max(x1_start, x2_start)
    inter_rect_y1 = max(y1_start, y2_start)
    inter_rect_x2 = min(x1_end, x2_end)
    inter_rect_y2 = min(y1_end, y2_end)

    if inter_rect_x1 < inter_rect_x2 and inter_rect_y1 < inter_rect_y2:
        intersection_area = (inter_rect_x2 - inter_rect_x1) * (inter_rect_y2 - inter_rect_y1)
        rect1_area = (x1_end - x1_start) * (y1_end - y1_start)
        rect2_area = (x2_end - x2_start) * (y2_end - y2_start)

        iou = intersection_area / float(rect1_area + rect2_area - intersection_area)
        return iou > 0.3  # 设定IoU阈值
    return False


def cut_images(image, rects, confidences=None):
    """
    裁剪图片
    :param image:  图片
    :param rects:  矩形框列表
    :param confidences:  对应每个矩形框的置信度列表
    :return:  裁剪后的图片列表
    """
    result = []
    if confidences is not None:
        for (startX, startY, endX, endY), confidence in zip(rects, confidences):
            if confidence > 0.1:  # 设定一个阈值
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cropped = image[startY:endY, startX:endX]
                result.append(cropped)
        return result
    else:
        for (startX, startY, endX, endY) in rects:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cropped = image[startY:endY, startX:endX]
            result.append(cropped)
        return result


def ocr_images(cropped_images):
    if isinstance(cropped_images, list):
        text = []
        for cropped in cropped_images:
            text.append(pytesseract.image_to_string(cropped ))
        return text
    else:
        text = pytesseract.image_to_string(cropped_images)
        return text


def main_extract_by_east():
    model_path = "frozen_east_text_detection.pb"
    image_path = "test.jpg"

    net = load_model(model_path)
    image, blob = preprocess_image(image_path)
    if image is None:
        return
    scores, geometry = detect_by_east(net, blob)
    rects, confidences = extract_areas(scores, geometry)

    filtered_rects = filter_duplicates(rects, confidences)
    image = draw_rectangles(image, filtered_rects)

    cropped_images = cut_images(image, filtered_rects)

    text = ocr_images(cropped_images)

    cv2.imshow("Text Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for i in range(len(cropped_images)):
    #     cv2.imshow("Cropped Text", cropped_images[i])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    print(text)


# def main_extract_by_tesseraact():
#     image_path = "test.jpg"
#     image = cv2.imread(image_path)
#     if image is None:
#         return
#     boxes = detect_by_tesseraact(image)
#     hImg, wImg, _ = image.shape
#     for box in boxes.splitlines():
#         box = box.split(' ')
#         x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
#         cv2.rectangle(image,(x,hImg-y),(w))
#     cv2.imshow("Text Detection", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    main_extract_by_east()
    # main_extract_by_tesseraact()
