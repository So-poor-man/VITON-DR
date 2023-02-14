import cv2
import numpy as np
import scipy.io as sio

def get_cloth_data(Image):
    blurred = cv2.GaussianBlur(Image, (7, 7), 0)
    gaussImg = cv2.Canny(blurred, 20, 20)
    W, H = gaussImg.shape
    list_i = []
    list_j = []
    p = 0
    p_1 = 0
    img_zeros = np.zeros((256, 192), np.uint8)
    for i in range(W):
        for j in range(H):
            if gaussImg[i][j] == 255 and p % 15 == 0:
                img_zeros[i][j] = gaussImg[i][j]
                list_i.append((-i + 255) / 255)
                list_j.append((j) / 191)
                p_1 += 1  # 降采样后点数标记位
            p += 1  # 总点数标记位
    Keypoint = np.array(list(zip(list_j, list_i)))
    # print("方差：", np.var(Keypoint))
    file_name = ".\\models\\datamat\\data1.mat"
    data = sio.loadmat(file_name)
    data["x1"] = Keypoint
    sio.savemat(".\\models\\datamat\\data1.mat", data)

def get_mask_data(Image):
    blurred = cv2.GaussianBlur(Image, (7, 7), 0)
    gaussImg = cv2.Canny(blurred, 20, 20)
    W, H = gaussImg.shape
    list_i = []
    list_j = []
    p = 0
    p_1 = 0
    img_zeros = np.zeros((256, 192), np.uint8)
    for i in range(W):
        for j in range(H):
            if gaussImg[i][j] == 255 and p % 5 == 0:
                img_zeros[i][j] = gaussImg[i][j]
                list_i.append((-i + 255) / 255)
                list_j.append((j) / 191)
                p_1 += 1  # 降采样后点数标记位
            p += 1  # 总点数标记位
    Keypoint = np.array(list(zip(list_j, list_i)))
    # print("方差：", np.var(Keypoint))
    file_name = ".\\models\\datamat\\data1.mat"
    data = sio.loadmat(file_name)
    data["y2a"] = Keypoint
    sio.savemat(".\\models\\datamat\\data1.mat", data)