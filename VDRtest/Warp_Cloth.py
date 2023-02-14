import cv2
import numpy as np
from skimage import exposure
def get_warp_image_channel(image, mask_cloth):
    key_point_path = ".//transform_cloth//txtdata//data.txt"
    point = get_point_data(key_point_path)
    W, H, _ = image.shape
    B, G, R = cv2.split(image)
    warp_R = np.zeros((W, H), np.uint8)
    warp_G = np.zeros((W, H), np.uint8)
    warp_B = np.zeros((W, H), np.uint8)
    warp_R,warp_G,warp_B = cloth_to_warp(R, G, B, warp_R, warp_G, warp_B, point, mask_cloth)

    warp_merged = cv2.merge([warp_R, warp_G, warp_B])
    warp_merged = cv2.medianBlur(warp_merged, 5)
    warp_merged = exposure.adjust_gamma(warp_merged, 1.5)
    return warp_merged
def get_point_data(key_point_path):
    point = []
    point_temp = []
    with open(key_point_path, 'r+', encoding='utf-8') as f:
        s = [i[:-1].split(',') for i in f.readlines()]
        for i in s:
            for j in i:
                if j =='':
                    break
                # point_temp.append(float(j))
                point_temp.append(int(round(float(j), 0)))
        point = np.vstack(point_temp)
    point = np.array(point).reshape(-1, 4)
    return point
def cloth_to_warp(R,G,B, warp_R, warp_G, warp_B, point, mask_cloth):
    N, _ = point.shape
    W, H = mask_cloth.shape
    point_distence = []
    for i in range(W):
        for j in range(H):
            if mask_cloth[i][j] > 220: #利用衣服掩码判断彩色衣服所在的位置
                K = np.array([j, i])
                K = np.tile(K,(N, 1))
                point_distence_temp = K - point[:,0:2]
                point_distence = point_distence_temp[:,0] ** 2 + point_distence_temp[:, 1] ** 2

                flag = point_distence.argmin()
                # for k in point: # 读取数据表中的每一行数据
                #     """
                #     注意：matlab中的读取数据的顺序与python中的读取顺序不同，主要原因是表达数据的坐标系不同，python中坐标原点在左上方，matlab在左下方，python中的（x/255, y/191）
                #     要转为（y/191, (-x+255)/255)输入到matlab
                #     """
                #     point_distence_temp = math.sqrt((k[1] - i)**2 + (k[0] - j)**2)#计算某个像素点到采样点的距离
                #     point_distence.append(point_distence_temp)
                # flag = point_distence.index(min(point_distence))
                #像素点位置偏移
                i_temp = i+point[flag][3]
                j_temp = j+point[flag][2]
                if (i_temp >0 & i_temp <255) and (j_temp > 0 & j_temp < 192):
                #像素点赋值
                    warp_R[i_temp][j_temp] = R[i][j]
                    warp_G[i_temp][j_temp] = G[i][j]
                    warp_B[i_temp][j_temp] = B[i][j]
                else:
                    pass
                # point_distence.clear()
    return warp_R, warp_G, warp_B