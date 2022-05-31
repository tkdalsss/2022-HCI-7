import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

orange = cv2.imread('./orange.jpg')
orange = cv2.resize(orange, (512, 512))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    cv2.imshow('video', img)

    faces = detector(img)

    result = orange.copy()

    if len(faces) > 0:
        face = faces[0]

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy()

        shape = predictor(img, face)
        shape = face_utils.shape_to_np(shape)
        
        # shape에 얼굴 점마다 좌표가 찍히는것!
        # # shape값 디버깅..
        # print("#######################")
        # print(shape)
        # print(len(shape)) # 68개 나온다..랜드마크 0 ~ 67인듯
        # print("@@@@@@@@@@@@@@@@@@@@@@@")
        # shape[landmark_number, x(0) or y(1)]

        for p in shape:
            cv2.circle(face_img, (p[0] - x1, p[1] - y1), 2, 255, -1)

        # eyes
        left_x1 = shape[36, 0]
        left_y1 = shape[37, 1]
        left_x2 = shape[39, 0]
        left_y2 = shape[41, 1]
        left_margin = int((left_x2 - left_x1) * 0.18)

        right_x1 = shape[42, 0]
        right_y1 = shape[43, 1]
        right_x2 = shape[45, 0]
        right_y2 = shape[47, 1]
        right_margin = int((right_x2 - right_x1) * 0.18)

        left_eye = img[left_y1 - left_margin:left_y2 + left_margin, left_x1 - left_margin:left_x2 + left_margin].copy()
        right_eye = img[right_y1 - right_margin:right_y2 + right_margin, right_x1 - right_margin:right_x2 + right_margin].copy()

        left_eye = resize(left_eye, 100)
        right_eye = resize(right_eye, 100)

        result = cv2.seamlessClone(
            left_eye,
            result,
            np.full(left_eye.shape[:2], 255, left_eye.dtype),
            (160, 200),
            cv2.MIXED_CLONE
        )

        result = cv2.seamlessClone(
            right_eye,
            result,
            np.full(right_eye.shape[:2], 255, right_eye.dtype),
            (350, 200),
            cv2.MIXED_CLONE
        )

        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        mouth_img = img[mouth_y1 - mouth_margin:mouth_y2 + mouth_margin, mouth_x1 - mouth_margin:mouth_x2 + mouth_margin].copy()

        mouth_img = resize(mouth_img, 250)

        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (250, 350),
            cv2.MIXED_CLONE
        )
        
        # face line 구현한것
        faceLine_x1 = shape[0, 0] #왼쪽 끝 x좌표
        faceLine_y1 = shape[27, 1]#미간 y좌표
        faceLine_x2 = shape[16, 0]#오른쪽 끝 x좌표
        faceLine_y2 = shape[8, 1] #턱 y좌표
        faceLine_margin = int((faceLine_x2 - faceLine_x1) * 0.02)
              
        faceLine_img = img[faceLine_y1 - faceLine_margin:faceLine_y2 + faceLine_margin, faceLine_x1 - faceLine_margin:faceLine_x2 + faceLine_margin].copy()
        faceLine_img = resize(faceLine_img, 300)  

        # face shape classification
        # 얼굴 너비
        face_left_x = shape[0, 0]
        face_left_y = shape[0, 1]
        face_right_x = shape[16, 0]
        face_right_y = shape[16, 1]
        
        #너비 계산
        tmp_face_width = (face_left_x-face_right_x)**2 + (face_left_y-face_right_y)**2
        face_width = math.sqrt(tmp_face_width);
        
        # 얼굴 길이
        face_top_x = shape[27, 0]
        face_top_y = shape[27, 1]
        face_bottom_x = shape[8, 0]
        face_bottom_y = shape[8, 1]        
        
        #길이 계산
        tmp_face_height = (face_top_x-face_bottom_x)**2 + (face_top_y-face_bottom_y)**2
        face_height = math.sqrt(tmp_face_height);
        
        #결과 출력
        print("----> User Face Width  =  {}" .format(face_width))
        print("----> User Face Height =  {}" .format(face_height))

        cv2.imshow('left', left_eye)
        cv2.imshow('right', right_eye)
        cv2.imshow('mouth', mouth_img)
        cv2.imshow('face', face_img)
        cv2.imshow('result', result)
        cv2.imshow('faceLine', faceLine_img)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("Your face info is being saved...")
            cv2.imwrite('result/lefteye{}.png'.format(1), left_eye, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite('result/righteye{}.png'.format(1), right_eye, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite('result/mouth{}.png'.format(1), mouth_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite('result/faceline{}.png'.format(1), face_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        elif key == ord('q'):
            break

        else:
            print("press key")
            break
