import requests
import json
import cv2
import time
from datetime import datetime

server = 'https://subeye.herokuapp.com/singleFile'
c_num=1

def send_img(img,g_num,c_num):
    if True:
        tt=time.strftime('%Y-%m-%d', time.localtime(time.time()))
        s = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        binary_cv = cv2.imencode('.png', img)[1].tobytes()

        upload = {'upload': binary_cv}
        data = {
            'g_num': g_num,
            'c_num': c_num,
            'createAt': s
        }
        try:
            res = requests.post(server, files=upload, data=data)
            if res.status_code == 200:
                print("Success!!")
                print(data)
        except RuntimeError as ex:
            print("Except!")
        print(res.status_code)
