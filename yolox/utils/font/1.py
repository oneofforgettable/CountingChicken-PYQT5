import cv2
import numpy as np
import matplotlib.font_manager as fm

# 加载中文字体
font_path = '<中文字体文件路径>'
font_prop = fm.FontProperties(fname=font_path)

# 创建一个空白图像
image = np.zeros((400, 600, 3), dtype=np.uint8)
image.fill(255)  # 白色背景

# 绘制汉字文本
text = '你好，世界！'
font_size = 40
text_color = (0, 0, 0)  # 黑色
text_position = (50, 200)  # 文本位置

cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
            font_size, text_color, thickness=2, lineType=cv2.LINE_AA,
            fontProperties=font_prop)

# 显示图像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()