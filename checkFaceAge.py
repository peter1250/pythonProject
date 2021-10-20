import cv2, glob, dlib
def face_age(img):
    age_list = [2, 6, 12, 20, 32, 43, 53, 80]

    detector = dlib.get_frontal_face_detector()
    age_net = cv2.dnn.readNetFromCaffe(
        'models/deploy_age.prototxt',
        'models/age_net.caffemodel')
    # cv2.imshow("ss",img)
    # cv2.waitKey(0)
    faces = detector(img)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy()
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=3, size=(227, 227),
                                     mean=(78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False, crop=False)
        print("얼굴")
        # predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # visualize
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        overlay_text = ' %s' % (age)
        cv2.putText(img, overlay_text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0, 0, 0), thickness=10)
        cv2.putText(img, overlay_text, org=(x1, y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        return age
    return -1
    # if cv2.waitKey(1) == ord('q'):
    #     break
