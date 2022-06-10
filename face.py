import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
import math
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
emoji_path = './data/emoji_base/m'

# emoji_base = cv2.resize(emoji_base, (512, 512))

cap = cv2.VideoCapture(0)
# 모폴로지 연산을 위한 SE

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
while cap.isOpened():
    '''
    ret, img = cap.read()

    if not ret:
        break
    '''
    img= cv2.imread('./data/photos/32.jpg')

    oh = img.shape[0]
    ow = img.shape[1]

    rw = 512
    rh = int((rw*oh)/ow)
    img_resized = cv2.resize(img, dsize=(rw,rh),interpolation= cv2.INTER_LINEAR)
    cv2.imshow('video', img_resized)
    faces = detector(img_resized)

    #result = emoji_base.copy()

    if len(faces) > 0:
        face = faces[0]

        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img_resized[y1:y2, x1:x2].copy()



        shape = predictor(img_resized, face)
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

        left_eye = img_resized[left_y1 - left_margin:left_y2 + left_margin, left_x1 - left_margin:left_x2 + left_margin].copy()
        right_eye = img_resized[right_y1 - right_margin:right_y2 + right_margin,
                    right_x1 - right_margin:right_x2 + right_margin].copy()

        left_eye = resize(left_eye, 100)
        right_eye = resize(right_eye, 100)


        left_eye_sharpened =cv2.filter2D(left_eye, -1, kernel)
        right_eye_sharpened = cv2.filter2D(right_eye, -1, kernel)



        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        mouth_img = img_resized[mouth_y1 - mouth_margin:mouth_y2 + mouth_margin,
                    mouth_x1 - mouth_margin:mouth_x2 + mouth_margin].copy()

        mouth_img = resize(mouth_img, 150)
        mouth_sharpened =cv2.filter2D(mouth_img, -1, kernel)



        # face line 구현한것
        faceLine_x1 = shape[0, 0]  # 왼쪽 끝 x좌표
        faceLine_y1 = shape[27, 1]  # 미간 y좌표
        faceLine_x2 = shape[16, 0]  # 오른쪽 끝 x좌표
        faceLine_y2 = shape[8, 1]  # 턱 y좌표
        faceLine_margin = int((faceLine_x2 - faceLine_x1) * 0.02)



        faceLine_img = img_resized[faceLine_y1 - faceLine_margin:faceLine_y2 + faceLine_margin,
                       faceLine_x1 - faceLine_margin:faceLine_x2 + faceLine_margin].copy()

        faceLine_img = resize(faceLine_img, 300)

        #원활한 이미지 색상 처리를 위해 hsv로 변환
        img_hsv =cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        cv2.imshow('img_hsv',img_hsv)
        img_h, img_s, img_v = cv2.split(img_hsv)
        cv2.imshow('img_h',img_h)
        cv2.imshow('img_s',img_s)
        cv2.imshow('img_v',img_v)

        #피부 색깔 구하기
        only_face =img_hsv[faceLine_y1 + int((faceLine_x2 - faceLine_x1) * 0.02):faceLine_y2 - int((faceLine_x2 - faceLine_x1) * 0.2),
                       faceLine_x1 + int((faceLine_x2 - faceLine_x1) * 0.1):faceLine_x2 - int((faceLine_x2 - faceLine_x1) * 0.1)].copy()

        cv2.imshow('only',only_face)
        h,s,v =cv2.split(only_face)
        median =np.array([np.percentile(h,55),np.percentile(s,55),np.percentile(v,55)]) #분위수 계산을 통해 피부색의 hsv를 구함
        #median = np.array([np.mean(b), np.mean(g), np.mean(r)])
        #print(median)

        #이마 영역 계산

        upperHead_x1 = faceLine_x1 - int((faceLine_x2-faceLine_x1) * 0.25)
        upperHead_x2 = faceLine_x2 + int((faceLine_x2-faceLine_x1) * 0.25)
        upperHead_y1 = faceLine_y1 - (faceLine_y2 - faceLine_y1)
        upperHead_y2 = faceLine_y1 - int((faceLine_y2 - faceLine_y1) * 0.05)
        '''
        upperHead_hsv = img_hsv[upperHead_y1-(upperHead_y2-upperHead_y1): upperHead_y1 - int((faceLine_y2 - faceLine_y1) * 0.05) ,
                       upperHead_x1 - int((upperHead_x2-upperHead_x1) * 0.25) : upperHead_x2 +  int((upperHead_x2-upperHead_x1) * 0.25)].copy()
        '''
        if upperHead_y1 < 0:
            upperHead_y1 =0
        upperHead_hsv = img_hsv[upperHead_y1:upperHead_y2,upperHead_x1:upperHead_x2].copy()
        cv2.imshow('upperHead_hsv', upperHead_hsv)

        #마스크의 영역의 아랫값
        bottom = np.array([median[0] - 10 , median[1] - 70, median[2] - 100])
        #흰색 배경까지 인식되는 것을 막기 위해
        if median[0] - 8 <= 0:
            bottom[0] =1
        #마스크의 영역의 윗값
        top = np.array([median[0] + 15, median[1] + 70, median[2] + 70])

        mask = cv2.inRange(upperHead_hsv, bottom, top) #이마의 피부 영역만 추출하기 위한 마스크

        upperHead_bgr = cv2.cvtColor(cv2.bitwise_and(upperHead_hsv, upperHead_hsv, mask=mask), cv2.COLOR_HSV2BGR) #해당 영역에 포함되는 부분만 남기고 모두 0 값으로 변환 배경부가 살색과 유사하면 효과가 떨어진다./
        cv2.imshow('upperHead_bgr', upperHead_bgr)
        upperHead_gray = cv2.cvtColor(upperHead_bgr, cv2.COLOR_BGR2GRAY)
        #모폴로지 연산을 통해 화이트 노이즈와 블랙 노이즈를 제거
        upperHead_gray = cv2.morphologyEx(upperHead_gray, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        upperHead_gray = cv2.morphologyEx(upperHead_gray, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


        '''
        upperHead_gray = cv2.cvtColor(upperHead2,cv2.COLOR_BGR2GRAY)
        ret, upperHead_bz= cv2.threshold(cv2.cvtColor(upperHead_hsv, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        upperHead_gray = cv2.morphologyEx(upperHead_gray, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        upperHead_gray = cv2.morphologyEx(upperHead_gray, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        '''
        #이마의 피부 영역과 upper_head와 피부 영역의 비율을 계산
        skin_area = 0
        skin_ratio =float(0)
        cv2.imshow('upperHead_gray', upperHead_gray)
        for r in upperHead_gray:
            skin_area += np.count_nonzero(r)
        upperHead_true_area =(faceLine_y2 - faceLine_y1-int((faceLine_y2 - faceLine_y1) * 0.05))*(faceLine_x2-faceLine_x1 + 2*int((faceLine_x2-faceLine_x1) * 0.25))
        skin_ratio = skin_area/upperHead_true_area
        #print(upperHead_true_area)
        print("----> User skin area  =  {}".format(skin_area))
        print("----> User skin ratio  =  {}".format(skin_ratio))

        # face shape classification
        # 얼굴 너비
        face_left_x = shape[0, 0]
        face_left_y = shape[0, 1]
        face_right_x = shape[16, 0]
        face_right_y = shape[16, 1]

        # 너비 계산
        tmp_face_width = (face_left_x - face_right_x) ** 2 + (face_left_y - face_right_y) ** 2
        face_width = math.sqrt(tmp_face_width)

        # 얼굴 길이
        face_top_x = shape[27, 0]
        face_top_y = shape[27, 1]
        face_bottom_x = shape[8, 0]
        face_bottom_y = shape[8, 1]

        # 길이 계산
        tmp_face_height = (face_top_x - face_bottom_x) ** 2 + (face_top_y - face_bottom_y) ** 2
        face_height = math.sqrt(tmp_face_height)
        # 턱의 각도 구하기 왼쪽 턱 3 4 5, 오른쪽 턱 11 12 13
        left_top_x = shape[3, 0]
        left_top_y = shape[3, 1]

        left_mid_x = shape[4, 0]
        left_mid_y = shape[4, 1]

        left_bottom_x = shape[5, 0]
        left_bottom_y = shape[5, 1]

        right_top_x = shape[13, 0]
        right_top_y = shape[13, 1]

        right_mid_x = shape[12, 0]
        right_mid_y = shape[12, 1]

        right_bottom_x = shape[11, 0]
        right_bottom_y = shape[11, 1]


        def angle(top_x, top_y, mid_x, mid_y, bot_x, bot_y):
            topLineSlope = (mid_y - top_y) / (mid_x - top_x)  # 기울기1 구하기
            botLineSlope = (mid_y - bot_y) / (mid_x - bot_x)  # 기울기2 구하기
            x = (abs(topLineSlope - botLineSlope)) / (1 + topLineSlope * botLineSlope)  # x 구하기
            angle_rad = np.arctan(x)  # x의 arctan값 구하기 / 구하면 radian값이 나옴
            angle_deg = np.degrees(angle_rad)  # radian값을 degree값으로 바꿔줌
            if angle_deg < 90:  # 교차각이 2개 나오므로, 90도보다 작은 각이 나오면 180도에서 이를 뺀 각을 교차각 최종값으로 정의
                angle_result = 180 - angle_deg
            else:
                angle_result = angle_deg
            return angle_result


        left_angle = angle(left_top_x, left_top_y, left_mid_x, left_mid_y, left_bottom_x, left_bottom_y)
        right_angle = angle(right_top_x, right_top_y, right_mid_x, right_mid_y, right_bottom_x, right_bottom_y)
        avg_angle = (left_angle + right_angle) / 2

        #이모지 생성부
        if skin_ratio < 0.1:
            emoji_path +='32.png'
        elif skin_ratio <0.2:
            emoji_path += '22.png'
        else:
            emoji_path += '12.png'

        emoji_base = cv2.imread(emoji_path)
        result = emoji_base.copy()
        result = cv2.seamlessClone(
            left_eye_sharpened,
            result,
            np.full(left_eye_sharpened.shape[:2], 255, left_eye_sharpened.dtype),
            (300, 400),
            cv2.NORMAL_CLONE
            # cv2.MIXED_CLONE
        )

        result = cv2.seamlessClone(
            right_eye_sharpened,
            result,
            np.full(right_eye_sharpened.shape[:2], 255, right_eye_sharpened.dtype),
            (455, 400),
            cv2.NORMAL_CLONE
            # cv2.MIXED_CLONE
        )
        result = cv2.seamlessClone(
            mouth_sharpened,
            result,
            np.full(mouth_sharpened.shape[:2], 255, mouth_sharpened.dtype),
            (385, 550),
            # cv2.MIXED_CLONE
            cv2.NORMAL_CLONE
        )

        # 결과 출력
        print("----> User Face Width  =  {}".format(face_width))
        print("----> User Face Height =  {}".format(face_height))
        print("----> User Face Angle =  {}".format(avg_angle))

        cv2.imshow('left', left_eye)
        cv2.imshow('right', right_eye)
        cv2.imshow('mouth', mouth_img)
        cv2.imshow('face', face_img)
        cv2.imshow('result', result)
        cv2.imshow('faceLine', faceLine_img)

        cv2.waitKey()
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("Your face info is being saved...")
            cv2.imwrite('result/lefteye{}.png'.format(1), left_eye, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite('result/righteye{}.png'.format(1), right_eye, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite('result/mouth{}.png'.format(1), mouth_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite('result/faceline{}.png'.format(1), face_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        elif key == ord('c'):
            break

        elif key == ord('q'):
            break

        else:
            print("press key")
            break
