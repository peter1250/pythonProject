import numpy as np
import torch
import pandas as pd
import cv2
import tqdm
import torchvision
import matplotlib
import seaborn
from PIL import Image
import Find_UseCard
import checkFaceAge
import sendIMG


class detect_gate:
    df=0
    img_show=0
    gate = []
    blackBox=[]

    def start_detect(self,results):
        return self.change_df(results)

    def change_df(self,results):
        df = pd.DataFrame(results.pandas().xyxy[0])
        df = df.sort_values('xmin')
        df = df.sort_values('class')
        self.df = df.reset_index(drop=True)
        return self.count_gate()

    def count_gate(self):
        countGate = int(self.df['class'].value_counts()[0:1].array[0])  # 게이트 갯수
        countBox = int(self.df['class'].value_counts()[1:2].array[0])  # 검은박스표지판 갯수
        return self.check_gate(countGate, countBox)

    def check_gate(self,countGate, countBox):
        good = 0
        for i in range(countGate):
            for j in range(countBox):
                if (self.df.loc[j + countGate, 'xmin'] > self.df.loc[i, 'xmin'] and self.df.loc[j + countGate, 'xmin'] < self.df.loc[
                    i, 'xmax']):
                    good += 1
                    if i + 1 < countGate:  # 다음 게이트가 있다면 실행
                        self.gate.append([(self.df.loc[i, 'xmin'] + self.df.loc[i, 'xmax']) / 2,
                                     (self.df.loc[i + 1, 'xmin'] + self.df.loc[i + 1, 'xmax']) / 2])  # 게이트 영역 좌표 리스트
                    else:
                        self.gate.append([(self.df.loc[i, 'xmin'] + self.df.loc[i, 'xmax']) / 2, 0])  # 마지막 게이트에서 사진 끝까지 표시

        if good == countBox:
            return self.check_blockBox(countGate, countBox)

    def check_blockBox(self,countGate,countBox):
        for i in range(countBox):
            xmax = int(self.df['xmax'][countGate + i])  # 오른 끝점
            xmin = int(self.df['xmin'][countGate + i])  # 왼 끝점
            ymax = int(self.df['ymax'][countGate + i])  # 아래 끝점
            ymin = int(self.df['ymin'][countGate + i])  # 위 끝점
            y = int((ymax - ymin) // 3 * 2.2 + ymin)  # 특별 승차자 박스 위 좌표
            self.blackBox.append([y, ymax, xmin, xmax])
        return True

def arrest_ticket(videoset):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
    model.conf = 0.8

    # 이미지 불러오기

    cap = cv2.VideoCapture(videoset)

    success = False

    while cap.isOpened():  # 게이트 추적 반복
        ret, img = cap.read()

        if not ret:
            print("실패")
        # img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        dg = detect_gate()  # 게이트 추척 클래스 지정
        success = dg.start_detect(model(img[..., ::-1], size=640))  # 추적 시작 성공시 True 반환
        print("탐색중")

        if success:
            break

    UseCard = Find_UseCard.UseCard(dg.blackBox)

    lightOn = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    print(dg.gate)
    ages = [0, 0, 0, 0, 0, 0, 0, 0]
    ageCount = [0, 0, 0, 0, 0, 0, 0]

    while cap.isOpened():  # 게이트 추적 반복
        ret, img = cap.read()
        # img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        if not ret:
            print("실패")

        for i in range(len(dg.blackBox)):
            checklight = lightOn[i][0]
            lightOn[i][0] = UseCard.Find_Kid(img, i)
            print(lightOn[i][0])
            if checklight != lightOn[i][0]:
                lightOn[i][1] = 10

        for i in range(len(dg.blackBox)):
            if lightOn[i][0] > 0 and lightOn[i][1] >= 0:
                lightOn[i][1] -= 1
                roi = img[0:, int(dg.gate[i][0]):int(dg.gate[i][1])]
                age = checkFaceAge.face_age(roi)
                print("추적시작")
                if age > 0:
                    ages[i] = ages[i] + int(age)
                    ageCount[i] += 1
                    print(age)

            if lightOn[i][1] == -1:
                lightOn[i][1] = -2
                if ages[i] / 10 > 13:
                    print(ages[i] / ageCount[i])
                    ageCount[i] = 0
                    sendIMG.send_img(roi, i + 1, 1)

        cv2.namedWindow("gate", flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("gate", 1000, 1000)
        cv2.imshow("gate", img)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyWindow()


