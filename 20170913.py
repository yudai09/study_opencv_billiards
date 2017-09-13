# -*- coding: utf-8

import cv2


def main():
    img = cv2.imread('./photos/IMG_1013.jpg')

    img = resize_image(img, 4)
    img_org = img.copy()

    cv2.imshow('image', img)
    cv2.waitKey(0)

    contors = find_contours(img)
    circles = find_circles(img)

    # 一番大きな輪郭がビリヤードテーブル（のはず
    contor_billiard_table = contors[0]

    # ビリヤード台を囲む四角
    # TODO: 台形補正する必要がある
    billiard_table_box = cv2.boundingRect(contor_billiard_table)

    print billiard_table_box

    x, y, w, h = billiard_table_box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for x, y, r in circles[0]:
        # img = img_org.copy()
        print (x, y, r)
        # x, y, r = int(x), int(y), int(r)
        cv2.circle(img, (x, y), r, (255, 255, 0), 4)

    cv2.imshow('image', img)
    cv2.waitKey(0)


def find_contours(img):
    #グレースケール
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # バイラテラルフィルタ
    img = cv2.bilateralFilter(img, 0, 24, 2)

    # Canny
    img = cv2.Canny(img, 50, 120)

    # 輪郭検出
    _, contours, _ = cv2.findContours(img,
                                      cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を面積が大きい順に並べ替える。
    contours.sort(key=cv2.contourArea, reverse=True)

    # for i, cnt in enumerate(contours):
    #     # 輪郭描画
    #     cv2.drawContours(img, contours, i, (255,0,0), 3)

    #     # 輪郭を囲う長方形を描画する
    #     # 輪郭が台形になっていることがわかったりする
    #     x, y, w, h = cv2.boundingRect(contours[i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)
    #     if i >= 50:
    #         break;

    return contours



def find_circles(img):
    #グレースケール
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # バイラテラルフィルタ
    # このフィルタのかけ方が円の検出にも影響してくる。
    # フィルタが強いと円の縁がよわくなり、結果として玉が認識されない
    img = cv2.bilateralFilter(img, 0, 24, 2)

    # canny
    # cannyはHoughCirclesの中で実行される
    # ここでは参考までに同じパラメタでCannyを実行した場合の値をみる
    img_canny = cv2.Canny(img, 75, 150)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                               # param1の値は大きい方がよいっぽい
                               param1=200, param2=70,
                               dp=2, minDist=5, minRadius=1, maxRadius=100)

    if circles is None or len(circles) <= 0:
        raise Exception('cannot find any balls')

    # for x, y, r in circles[0]:
    #     # img = img_org.copy()
    #     print (x, y, r)
    #     # x, y, r = int(x), int(y), int(r)
    #     cv2.circle(img, (x, y), r, (255, 255, 0), 4)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    return circles

def resize_image(img, reduction_ratio):
    # resize
    height, width = img.shape[:2]
    size = (width/reduction_ratio, height/reduction_ratio)
    return cv2.resize(img, size)


if __name__ == '__main__':
    main()



