import numpy as np
import cv2

img_color = cv2.imread('img/hsv1.png')
# 이미지 파일을 컬러로 불러옴
print('shape: ', img_color.shape)
height, width = img_color.shape[:2]
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

lower_blue = (20, 150, 150)
# hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_blue = (30, 255, 255)
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask)

cv2.namedWindow("img_origin",flags=cv2.WINDOW_NORMAL)
cv2.namedWindow("img_mask",flags=cv2.WINDOW_NORMAL)
cv2.namedWindow("img_color",flags=cv2.WINDOW_NORMAL)

cv2.resizeWindow("img_origin",500,500)
cv2.resizeWindow("img_mask",500,500)
cv2.resizeWindow("img_color",500,500)

cv2.imshow("img_origin", img_color)
cv2.imshow("img_mask", img_mask)
cv2.imshow("img_color", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

pixels = cv2.countNonZero(img_mask )
if pixels > 1000:
    print("green exist")
    print(pixels )
else:
    print("not found")
