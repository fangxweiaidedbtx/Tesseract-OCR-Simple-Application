# 语言选择

[中文简体](readme.md)\\[English](readme_en.md)

# tesseract 文字识别简单应用

本程序是基于Python和pytesseract库实现的。主要功能是对图像中的文字进行识别，并按照指定的方式打印出来。优点如下：

1. 识别率较高：灰度化以及二值化后可以提高图像的质量和OCR识别率。

2. 打印方式灵活：可以按照“原始”或“按行”的方式打印识别到的文字。

3. 支持多图片识别：可以同时处理多个图片文件。

   

# 安装依赖

pip install -r requirements.txt

请自行安装tesseract并配置全局变量

链接如下：https://tesseract-ocr.github.io/tessdoc/Installation.html

# 运行

在main.py中更改图片路径，然后执行main.py即可



# 效果

输入图片

![1727271529335](C:\Users\19389\AppData\Local\Temp\1727271529335.png)

识别结果

![result](D:\code\ocr\result.png)

对英文支持更好，中文略有欠缺，不支持艺术字。





