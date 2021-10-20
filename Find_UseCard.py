import cv2
class UseCard:
    blackBoxs=[]
    def __init__(self,blackBoxs):
        self.blackBoxs=blackBoxs

    def Find_Kid(self,img,i):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = (10, 150, 150)
        upper_blue = (30, 255, 255)
        roi = hsv[self.blackBoxs[i][0]:self.blackBoxs[i][1], self.blackBoxs[i][2]:self.blackBoxs[i][3]]
        img_mask = cv2.inRange(roi, lower_blue, upper_blue)
        pixels = cv2.countNonZero(img_mask)
        if pixels > 10:
            print(pixels)
            return 1
        else:
            return 0





