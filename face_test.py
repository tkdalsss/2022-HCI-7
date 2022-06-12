import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
import math
import matplotlib.pyplot as plt
import angle
import time

def make_emoji_test(input_image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    emoji_path = './data/emoji_base/m'



    #샤프닝을 위한 하이패스 필터
    hp_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    img = cv2.imread(input_image_path)
    oh = img.shape[0]
    ow = img.shape[1]

    rw = 512
    rh = int((rw * oh) / ow)
    img_resized = cv2.resize(img, dsize=(rw, rh), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('video', img_resized)
    faces = detector(img_resized)
    dst = img_resized.copy()
    # result = emoji_base.copy()

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

        left_eye = img_resized[left_y1 - left_margin:left_y2 + left_margin,
                   left_x1 - left_margin:left_x2 + left_margin].copy()
        right_eye = img_resized[right_y1 - right_margin:right_y2 + right_margin,
                    right_x1 - right_margin:right_x2 + right_margin].copy()

        left_eye = resize(left_eye, 95)
        right_eye = resize(right_eye, 95)

       

        # mouth
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

        mouth_img = img_resized[mouth_y1 - mouth_margin:mouth_y2 + mouth_margin,
                    mouth_x1 - mouth_margin:mouth_x2 + mouth_margin].copy()

        mouth_img = resize(mouth_img, 150)
      

        # face line 구현한것
        faceLine_x1 = shape[0, 0]  # 왼쪽 끝 x좌표
        faceLine_y1 = shape[27, 1]  # 미간 y좌표
        faceLine_x2 = shape[16, 0]  # 오른쪽 끝 x좌표
        faceLine_y2 = shape[8, 1]  # 턱 y좌표
        faceLine_margin = int((faceLine_x2 - faceLine_x1) * 0.02)

        faceLine_img = img_resized[faceLine_y1 - faceLine_margin:faceLine_y2 + faceLine_margin,
                       faceLine_x1 - faceLine_margin:faceLine_x2 + faceLine_margin].copy()

        faceLine_img = resize(faceLine_img, 300)

        try:

            startA = time.time_ns()

            # 원활한 이미지 색상 처리를 위해 hsv로 변환
            img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

            # 피부 색깔 구하기
            only_face = img_hsv[faceLine_y1 + int((faceLine_x2 - faceLine_x1) * 0.02):faceLine_y2 - int(
                (faceLine_x2 - faceLine_x1) * 0.2),
                        faceLine_x1 + int((faceLine_x2 - faceLine_x1) * 0.1):faceLine_x2 - int(
                            (faceLine_x2 - faceLine_x1) * 0.1)].copy()

            # cv2.imshow('only', only_face)
            h, s, v = cv2.split(only_face)
            median = np.array(
                [np.percentile(h, 55), np.percentile(s, 55), np.percentile(v, 55)])  # 분위수 계산을 통해 피부색의 hsv를 구함
            # median = np.array([np.mean(b), np.mean(g), np.mean(r)])
            # print(median)

            # 이마 영역 계산

            upperHead_x1 = faceLine_x1 - int((faceLine_x2 - faceLine_x1) * 0.25)
            upperHead_x2 = faceLine_x2 + int((faceLine_x2 - faceLine_x1) * 0.25)
            upperHead_y1 = faceLine_y1 - (faceLine_y2 - faceLine_y1)
            upperHead_y2 = faceLine_y1 - int((faceLine_y2 - faceLine_y1) * 0.05)
            '''
            upperHead_hsv = img_hsv[upperHead_y1-(upperHead_y2-upperHead_y1): upperHead_y1 - int((faceLine_y2 - faceLine_y1) * 0.05) ,
                           upperHead_x1 - int((upperHead_x2-upperHead_x1) * 0.25) : upperHead_x2 +  int((upperHead_x2-upperHead_x1) * 0.25)].copy()
            '''
            if upperHead_y1 < 0:
                upperHead_y1 = 0
            upperHead_hsv = img_hsv[upperHead_y1:upperHead_y2, upperHead_x1:upperHead_x2].copy()

            # 마스크의 영역의 아랫값
            bottom = np.array([median[0] - 10, median[1] - 70, median[2] - 100])
            # 흰색 배경까지 인식되는 것을 막기 위해
            if median[0] - 8 <= 0:
                bottom[0] = 1
            # 마스크의 영역의 윗값
            top = np.array([median[0] + 15, median[1] + 70, median[2] + 70])

            mask = cv2.inRange(upperHead_hsv, bottom, top)  # 이마의 피부 영역만 추출하기 위한 마스크

            upperHead_bgr = cv2.cvtColor(cv2.bitwise_and(upperHead_hsv, upperHead_hsv, mask=mask),
                                         cv2.COLOR_HSV2BGR)  # 해당 영역에 포함되는 부분만 남기고 모두 0 값으로 변환 배경부가 살색과 유사하면 효과가 떨어진다./
            # cv2.imshow('upperHead_bgr', upperHead_bgr)
            upperHead_gray = cv2.cvtColor(upperHead_bgr, cv2.COLOR_BGR2GRAY)
            # 모폴로지 연산을 통해 화이트 노이즈와 블랙 노이즈를 제거
            upperHead_gray = cv2.morphologyEx(upperHead_gray, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            upperHead_gray = cv2.morphologyEx(upperHead_gray, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            # 이마의 피부 영역과 upper_head와 피부 영역의 비율을 계산
            skin_area = 0
            skin_ratio = float(0)

            for r in upperHead_gray:
                skin_area += np.count_nonzero(r)
            upperHead_true_area = (faceLine_y2 - faceLine_y1 - int((faceLine_y2 - faceLine_y1) * 0.05)) * (
                    faceLine_x2 - faceLine_x1 + 2 * int((faceLine_x2 - faceLine_x1) * 0.25))
            skin_ratio = skin_area / upperHead_true_area
            # print(upperHead_true_area)
            print("----> User skin area  =  {}".format(skin_area))
            print("----> User skin ratio  =  {}".format(skin_ratio))
            print("algorithm A time :", time.time_ns() - startA)  # 현재시각 - 시작시간 = 실행 시간
        except:
            print('!!!exception occur in algorithm A.!!!')

        try:
            startB = time.time_ns()

            # 턱을 기준 으로 아래는 버리기
            tuck = shape[8, 1]
            dst_crop = dst[0: tuck, :]

            # dst_crop = dst

            dst_crop_1 = detector(dst_crop)
            dst_crop_2 = dst_crop_1[0]

            shape2 = predictor(img, dst_crop_2)
            shape2 = face_utils.shape_to_np(shape2)

            x1, y1, x2, y2 = dst_crop_2.left(), dst_crop_2.top(), dst_crop_2.right(), dst_crop_2.bottom()
            dst_crop_img = img[y1:y2, x1:x2].copy()

            img = dst_crop

            # 얼굴 부분을 추출하기 위해 선을 그리기
            landmark_tuple = []
            for k, d in enumerate(dst_crop_1):
                landmarks = predictor(img, d)
                for n in range(0, 27):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmark_tuple.append((x, y))
                    cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

            routes = []

            for i in range(15, -1, -1):
                from_coordinate = landmark_tuple[i + 1]
                to_coordinate = landmark_tuple[i]
                routes.append(from_coordinate)

            from_coordinate = landmark_tuple[0]
            to_coordinate = landmark_tuple[17]
            routes.append(from_coordinate)

            for i in range(17, 20):
                from_coordinate = landmark_tuple[i]
                to_coordinate = landmark_tuple[i + 1]
                routes.append(from_coordinate)

            from_coordinate = landmark_tuple[19]
            to_coordinate = landmark_tuple[24]
            routes.append(from_coordinate)

            for i in range(24, 26):
                from_coordinate = landmark_tuple[i]
                to_coordinate = landmark_tuple[i + 1]
                routes.append(from_coordinate)

            from_coordinate = landmark_tuple[26]
            to_coordinate = landmark_tuple[16]
            routes.append(from_coordinate)
            routes.append(to_coordinate)

            for i in range(0, len(routes) - 1):
                from_coordinate = routes[i]
                to_coordinate = routes[i + 1]
                img = cv2.line(img, from_coordinate, to_coordinate, (255, 255, 0), 1)

            # for p in shape2:
            #    cv2.circle(img, (p[0] - x1, p[1] - y1), 2, 255, -1)

            # 얼굴만 따서 crop 이미지 에서 빼주기
            mask = np.zeros((img.shape[0], img.shape[1]))
            mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
            mask = mask.astype(np.bool)

            out = np.zeros_like(img)
            out[mask] = img[mask]

            sub_img = cv2.subtract(img, out)
            sub_img_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)

            # thresholding을 진행 하여 이진화 하고 면적 구하기
            ret, thr = cv2.threshold(sub_img_gray, 100, 255, cv2.THRESH_BINARY_INV)

            # 긴머리와 짧은 머리를 구별하기 위해 이진화된 이미지에서의 픽셀 값 차이를 이용
            pixel_cnt = 0
            b_height, b_width = thr.shape

            for i in range(0, b_height):
                for j in range(0, b_width):
                    if thr[i, j] == 255:
                        pixel_cnt = pixel_cnt + 1

            print("----> User skin area  =  {}".format(pixel_cnt))

            contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dst_crop, contours, -1, (0, 255, 255), 2)
            c0 = contours[-1]
            M = cv2.moments(c0)

            img_copy = dst_crop.copy()
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            leftmost = tuple(c0[c0[:, :, 0].argmin()][0])
            rightmost = tuple(c0[c0[:, :, 0].argmax()][0])
            topmost = tuple(c0[c0[:, :, 1].argmin()][0])
            bottommost = tuple(c0[c0[:, :, 1].argmax()][0])

            area = cv2.contourArea(c0)
            print('---->contour area reuslt: ', area)

            print("algorithm B time :", time.time_ns() - startB)  # 현재시각 - 시작시간 = 실행 시간

        except:
            print('!!!exception occur in algorithm B.!!!')
        print('-----------------------------------')



if __name__ == '__main__':
    print('test ml1')
    make_emoji_test('./data/photos/man/ml1.jpg')
    print('test ml2')
    make_emoji_test('./data/photos/man/ml2.jpg')
    print('test ml3')
    make_emoji_test('./data/photos/man/ml3.jpg')
    print('test 11')
    make_emoji_test('./data/photos/man/11.jpg')
    print('test 12')
    make_emoji_test('./data/photos/man/12.jpg')
    print('test 13')
    make_emoji_test('./data/photos/man/13.jpg')
    print('test 21')
    make_emoji_test('./data/photos/man/21.jpg')
    print('test 22')
    make_emoji_test('./data/photos/man/22.jpg')
    print('test 23')
    make_emoji_test('./data/photos/man/23.jpg')
    print('test 31')
    make_emoji_test('./data/photos/man/31.jpg')
    print('test 32')
    make_emoji_test('./data/photos/man/32.jpg')
    print('test 33')
    make_emoji_test('./data/photos/man/33.jpg')