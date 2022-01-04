import cv2
import numpy as np
from os import walk
# !wget  https://i.stack.imgur.com/sDQLM.png


def process(img):
    rows, cols = img.shape[0:2]


    # R1 = pose1[0:3, 0:3]
    # t1 = pose1[0:3, 3]
    #
    # R2 = pose2[0:3, 0:3]
    # t2 = pose2[0:3, 3]
    #
    # # https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes
    # R = np.matmul(np.linalg.inv(R2), R1)
    # T = np.matmul(np.linalg.inv(R2), (t1 - t2))
    R = np.eye(3)       # macierz obrotu
    T = np.zeros((3, 1))        # macierz przesuniecia


    # https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
    px = 320.0
    py = 320.0
    fx = 500.0
    fy = 500.0

    cameraMatrix1 = np.array(
        [
            [fx, 0, px],
            [0, fy, py],
            [0, 0, 1.0]
        ]
    )

    cameraMatrix2 = cameraMatrix1

    distCoeff = np.zeros(4)
    # distCoeff[0] = 0.002
    distCoeff[3] = -0.007

    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=cameraMatrix1,
        distCoeffs=distCoeff,
        R=R,
        newCameraMatrix=cameraMatrix1,
        size=(cols, rows),
        m1type=cv2.CV_32FC1)



    img_rect = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)




    # crop  <============================================3
    img_crop = img_rect[60:rows-95, 20:cols-20]         #tutaj do zmiany eksperymentalnie dla roznych folderow (czasem przstawiala sie kamera)

    img_gray = cv2.cvtColor(img_crop,cv2.COLOR_BGR2GRAY)

    # detect lines
    edges = cv2.Canny(img_gray,50,150,apertureSize = 3)
    cv2.imwrite('edges.jpg',edges)
    minLineLength=30
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=20,lines=np.array([]), minLineLength=minLineLength,maxLineGap=20)

    coords = []

    if lines is not None:
        # print("lines found")
        a,b,c = lines.shape
        for i in range(a):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]


            angle = np.arctan2(y2-y1, x2-x1) * 180. / np.pi

            if angle < 95 and angle > 85:
                coords.append(x1)
                coords.append(x2)
                # cv2.line(img_crop, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    else:
        return img_crop

    if not coords:
        return img_crop

    if len(coords) > 3:
        # crop to line bounds
        x_min = min(coords)
        x_max = max(coords)
        if(x_max - x_min > 100):
            rows, cols = img_crop.shape[0:2]

            img_crop = img_crop[0: rows, x_min+15:x_max-5]


    return img_crop

if __name__ == "__main__":
    mypath = 'I:\\GitHub\\Image-sharpening\\dataset\\test\\'      #zmienic sciezke do folderu
    # mypath = 'I:\\GitHub\\Image-sharpening\\dataset\\train\\'  # train albo test, wybrać jedno i zakomentowac
    filenames = next(walk(mypath + "blur"), (None, None, []))[2]  # [] if no file

    for filename in filenames:
        img = cv2.imread(mypath + "blur\\" + filename)
        img_processed = process(img)
        cv2.imwrite(mypath + "processed\\" + filename, img_processed)
