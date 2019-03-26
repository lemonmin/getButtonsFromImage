import numpy as np
import cv2, random
import pytesseract as teser

def contour():
    img = cv2.imread('test/captured_main_page.jpg')
    imgCopy = img.copy()
    # imgCopy = cv2.resize(imgCopy, None, fx=0.5, fy=0.5)
    imgray = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)
    img1 = imgCopy.copy()
    img2 = imgCopy.copy()

    ret, thr = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY) # cv2.threshold(img, threshold_value, value, flag)
    cv2.imshow('thr', thr)
    # [threshold_value] : 픽셀 문턱값
    #  [value] : 픽셀 문턱값보다 크거나 작을 때 적용되는 최대 값 (플래그에 따라 클 때일지 작을 때일지 정해짐)
    #  [flag] : 문턱값 적용 방법
    # cv2.THRESH_BINARY : 픽셀 값이 threshold_value 보다 크면 value, 작으면 0으로 할당
    _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp = []
    for i in range(len(contours)):
        print("currnet i =",i, end=" ** ")
        cnt = contours[i]
        area = cv2.contourArea(cnt) # 폐곡선으로 둘러쌓인 cnt의 면적 구하기
        print("area = ",area)
        if area < 1000:
            # if temp == []:
            #     temp = cnt
            # else:
            #     try:
            #         temp = np.vstack([temp, [cnt]])
            #     except Exception as err:
            #         temp = np.vstack([temp, cnt])
            continue
        else:
            # if temp != []:
            #     draw_cnt(temp, img1, img2, imgCopy)
            temp = cnt
        draw_cnt(temp, img1, img2, imgCopy)
        temp = []
    cv2.destroyAllWindows()

def draw_cnt(temp, img1, img2, imgCopy):
    cv2.drawContours(imgCopy, [temp], 0, (0, 0, 255), 2)

    epsilon1 = 0.03 * cv2.arcLength(temp, True) # epsilon 값으로 전체 둘레 길이의 3%를 할당
    epsilon2 = 0.1 * cv2.arcLength(temp, True) # epsilon 값으로 전체 둘레 길이의 10%를 할당

    approx1 = cv2.approxPolyDP(temp, epsilon1, True)
    approx2 = cv2.approxPolyDP(temp, epsilon2, True)

    cv2.drawContours(img1, [approx1], 0, (0, 0, 255), 2)
    cv2.drawContours(img2, [approx2], 0, (0, 0, 255), 2)

    # cv2.imshow('contour', imgCopy)
    cv2.imshow('Approx1', img1)
    # cv2.imshow('Approx2', img2)

    cv2.waitKey(0)

# r1 = (start p, end p)
# r2가 r1에 포함되는지
def check_contain(r1, r2):
    return r2[0][0]+2 >= r1[0][0] and r2[0][1]+2 >= r1[0][1] and r2[1][0]-2 <= r1[1][0] and r2[1][1]-2 <= r1[1][1]

def thr_grouping(imgCopy, thr):

    grouping_near_pixel = []

    for y in range(int(thr.shape[0])): # to 719
        toggle = False
        toggle_on_count = 0
        colmn = []
        start_x = 0
        end_x = 0
        for x in range(int(thr.shape[1])): # to 1279
            if toggle: # start가 있는 상황
                if x == 1279:
                    if thr[y][x] != 0:
                        if start_x != x:
                            colmn.append((start_x, x))
                    else:
                        if start_x != end_x:
                            colmn.append((start_x, end_x))
                elif toggle_on_count > 5:
                    if thr[y][x] != 0:
                        end_x = x
                        toggle_on_count = 0
                    else:
                        if start_x != end_x:
                            colmn.append((start_x, end_x))
                        toggle = False
                else:
                    if thr[y][x] != 0:
                        end_x = x
                        toggle_on_count = 0
                    else:
                        toggle_on_count += 1
            else:
                if thr[y][x] != 0:
                    start_x = x
                    end_x = x
                    toggle = True
        grouping_near_pixel.append(colmn)

    grouping_items = []

    for i in range(len(grouping_near_pixel)): # group 은 가로1줄의 start_x, end_x 뭉텅이들
        while True:
            if len(grouping_near_pixel[i]) < 1:
                break
            temp = grouping_near_pixel[i].pop(0)
            temp_start_point = [temp[0], i]
            temp_end_point = [temp[1], i]
            finish_count = 0


            for y in range(i, len(grouping_near_pixel)):
                idx = 0
                found = False
                while True:
                    if idx >= len(grouping_near_pixel[y]):
                        break
                    cur = grouping_near_pixel[y][idx]
                    if (cur[0] >= temp_start_point[0] and cur[0] <= temp_end_point[0]) or (cur[1] >= temp_start_point[0] and cur[1] <= temp_end_point[0]):
                    # if abs(cur[0] - temp_start_point[0]) < 5 and abs(cur[1] - temp_end_point[0]) < 5:
                        tmp = grouping_near_pixel[y].pop(idx)
                        x = temp_end_point[0]
                        if tmp[1] > x:
                            x = tmp[1]
                        temp_end_point = [x, y]
                        if tmp[0] < temp_start_point[0]:
                            temp_start_point = [tmp[0], temp_start_point[1]]
                        found = True
                    idx += 1
                if not found:
                    finish_count += 1
                if finish_count > 5:
                    break

            skip = False

            for i, item in enumerate(grouping_items):
                st_x, st_y = item[0]
                en_x, en_y = item[1]
                if check_contain(item, (temp_start_point, temp_end_point)):
                    skip = True
                    break
                if check_contain((temp_start_point, temp_end_point), item):
                    grouping_items[i] = [temp_start_point, temp_end_point]
                    skip = True
                    break
                if temp_start_point[0] >= st_x and temp_start_point[0] <= en_x and temp_start_point[1] >= st_y and temp_start_point[1] <= en_y:
                    if temp_end_point[0] > en_x:
                        grouping_items[i][1][0] = temp_end_point[0]
                    if temp_end_point[1] > en_y:
                        grouping_items[i][1][1] = temp_end_point[1]
                    skip = True
                    break
                if temp_end_point[0] >= st_x and temp_end_point[0] <= en_x and temp_end_point[1] >= st_y and temp_end_point[1] <= en_y:
                    if temp_start_point[0] < st_x:
                        grouping_items[i][0][0] = temp_start_point[0]
                    if temp_start_point[1] < st_y:
                        grouping_items[i][0][1] = temp_start_point[1]
                    skip = True
                    break
                if (st_y <= temp_start_point[1] <= en_y or st_y <= temp_end_point[1] <= en_y) and (temp_start_point[0] <= st_x <= temp_end_point[0] or temp_start_point[0] <= en_x <= temp_end_point[0]):
                    if temp_start_point[0] < st_x:
                        grouping_items[i][0][0] = temp_start_point[0]
                    if temp_start_point[1] < st_y:
                        grouping_items[i][0][1] = temp_start_point[1]
                    if temp_end_point[0] > en_x:
                        grouping_items[i][1][0] = temp_end_point[0]
                    if temp_end_point[1] > en_y:
                        grouping_items[i][1][1] = temp_end_point[1]
                    skip = True
                    break

            if not skip:
                # temp_start_point[0] -= 2
                # temp_start_point[1] -= 2
                # temp_end_point[0] += 2
                # temp_end_point[1] += 2
                grouping_items.append([temp_start_point, temp_end_point])


    print("========= grouping_items == ", grouping_items)


    # for start_point, end_point in grouping_items:
    #     try:
    #         # cv2.circle(imgCopy, tuple(start), 1, [0, 0, 255], -1)
    #         # cv2.circle(imgCopy, tuple(end), 1, [0, 0, 255], -1)
    #         cv2.rectangle(imgCopy, tuple(start_point), tuple(end_point), [random.randrange(0, 256),random.randrange(0, 256),random.randrange(0, 256)],2)
    #     except Exception as err:
    #         # print(err, "start, end ",(start, end))
    #         continue
    # # cv2.drawContours(imgCopy, [filtering], 0, (0, 0, 255), 2)
    # cv2.imshow('Approx2', imgCopy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for start_point, end_point in grouping_items:
        try:
            cur_crop = imgCopy[start_point[1]-30:end_point[1]+30, start_point[0]-30:end_point[0]+30]
            cur_crop_gray = cv2.cvtColor(cur_crop, cv2.COLOR_BGR2GRAY)
            _, cur_crop_thr = cv2.threshold(cur_crop_gray, 120, 255, cv2.THRESH_BINARY)
            boxes = teser.image_to_string(cur_crop_thr)
            # if not boxes:
            #     boxes = teser.image_to_boxes(cur_crop_thr)
            print(">>>>>>>>>>>>>>>>>>>> ",boxes)
            print("type = ",type(boxes))
            # for b in boxes.splitlines():
            #     b = b.split()
            #     # cv2.rectangle(imgCopy, ((int(b[1]), 720 - int(b[2]))), ((int(b[3]), 720 - int(b[4]))), (0, 255, 0), 2)
            #     print(">>>>>>>>>>>>>>>>>> ",b[0])
            # for line in items:
            #     ch, x1, y1, x2, y2, _ = line.split()
            #     print(ch, x1, y1, x2, y2)
            #     cv2.rectangle(imgCopy, (int(x1), int(y1)), (int(x2), int(y2)), [random.randrange(0, 256),random.randrange(0, 256),random.randrange(0, 256)],2)
            cv2.imshow('Approx2', cur_crop_thr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as err:
            # print(err, "start, end ",(start, end))
            continue
    # cv2.drawContours(imgCopy, [filtering], 0, (0, 0, 255), 2)


def get_all_words(thr, img):
    boxes = teser.image_to_boxes(thr)
    print(">>>>>>>>>>>>>>>>>>>> ",boxes)
    h, w, _ = img.shape
    for b in boxes.splitlines():
        b = b.split()
        print(">>>>>>>>>>>>> ", b[0])
        cv2.rectangle(img, ((int(b[1]), h - int(b[2]))), ((int(b[3]), h - int(b[4]))), (0, 255, 0), 1)
    # for line in items:
    #     ch, x1, y1, x2, y2, _ = line.split()
    #     print(ch, x1, y1, x2, y2)
    #     cv2.rectangle(imgCopy, (int(x1), int(y1)), (int(x2), int(y2)), [random.randrange(0, 256),random.randrange(0, 256),random.randrange(0, 256)],2)
        cv2.imshow('Approx2', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()



# contour()
img = cv2.imread('test/captured_main_page.jpg')
imgCopy = img.copy()
imgray = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(imgray, 120, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
thr_grouping(imgCopy, thr)
# get_all_words(thr, imgCopy)
