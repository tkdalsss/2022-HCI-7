import numpy as np
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