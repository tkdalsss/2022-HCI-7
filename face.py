import cv2
import dlib
from imutils import face_utils, resize
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('venv/Lib/shape_predictor_68_face_landmarks.dat')

orange = cv2.imread('C:\\Users\\lsm99\\Desktop\\orange.jpg')
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

        cv2.imshow('left', left_eye)
        cv2.imshow('right', right_eye)
        cv2.imshow('mouth', mouth_img)
        cv2.imshow('face', face_img)

        cv2.imshow('result', result)

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
