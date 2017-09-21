# -*- coding: utf-8

import cv2
import numpy as np


def main():
    images=[
        'photos/20170921/2017-09-21-185800.jpg',
        'photos/20170921/2017-09-21-185805.jpg',
        'photos/20170921/2017-09-21-185809.jpg',
        'photos/20170921/2017-09-21-185841.jpg',
        'photos/20170921/2017-09-21-185847.jpg',
        'photos/20170921/2017-09-21-185852.jpg',
    ]

    for img in images:
        img = cv2.imread(img)

        img_org = img.copy()

        # 表示
        cv2.imshow('original', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # バイラテラルフィルタ（平滑化）
        img = cv2.bilateralFilter(img, 0, 24, 2)

        # 表示
        cv2.imshow('bilateral filter', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # HSVカラー空間に変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 表示
        cv2.imshow('HSV', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # # 画像中央の色をサンプリングする
        # # 次のマスクに利用するため
        # # 値がわかればこの処理はコメントアウトしてもよい
        # height, width = img.shape[:2]
        # sampling_height = height / 8
        # sampling_width = width / 8

        # sampling_y0 = (height - sampling_height) / 2
        # sampling_x0 = (width - sampling_width) / 2

        # sum = [0, 0, 0]
        # max = [0, 0, 0]
        # min = [0, 0, 0]
        # for i in range(sampling_y0, sampling_y0 + sampling_height):
        #     for j in range(sampling_x0, sampling_x0 + sampling_width):
        #         # 標準出力にサンプリング領域の画像値を表示
        #         print(img[i][j])
        #         sum += img[i][j]
        #         img[i][j] = [0, 0, 0]

        # # 平均値を表示
        # print("average")
        # print(sum / (sampling_height * sampling_width))

        # # 表示
        # cv2.imshow('HSV-1', img)
        # cv2.waitKey(0)

        # 台だけを抽出するために色のマスクを作成する
        # ref. http://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
        greenLower = (40, 60, 40)
        greenUpper = (90, 255, 255)

	mask = cv2.inRange(img, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow('image', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 輪郭検出
	_, contours, _  = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		                          cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭を面積が大きい順に並べ替える。
        contours.sort(key=cv2.contourArea, reverse=True)

        cv2.drawContours(img, contours, 0, (255,0,0), 3)

        # 面積が一番大きい物を台とみなす
        pool = contours[0]

        cv2.imshow('pool', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ポケットで穴が開いているところを埋める
        # 埋めなくても大丈夫な気がするけど
        hull = cv2.convexHull(pool)
        cv2.drawContours(img, [hull], 0, (255,255,255), 3)

        # 黒いイメージに描画する
        height, width = img.shape[:2]
        img_solid_black = np.zeros((height, width, 3), np.uint8)
        img_hull = img_solid_black.copy()
        cv2.drawContours(img_hull, [hull], 0, (255,255,255), 3)

        cv2.imshow('hull', img_hull)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # グレースケール
        gray = cv2.cvtColor(img_hull, cv2.COLOR_BGR2GRAY)

        # Cannyでエッジ検出
        edges = cv2.Canny(gray,50,150,apertureSize = 3)

        # np.pi/360はnp.pi/180よりもよい結果を出す。分解能は高いほうがよい？？
        # ↑縦の線が180の時はしきい値を下げないと検出できない場合がある
        lines = cv2.HoughLines(edges, 1, np.pi/360, 250)

        # rhoでソートして似たような直線を配列上で近くに並ばせる
        lines = sorted(lines, key=lambda line: line[0][0])

        # 同じような直線同士の平均をとり一本の直線にまとめる
        lines_combined = []
        # 直線距離の差は10ピクセルまで
        max_diff_rho = 10
        # 角度の差は0.1度まで
        max_diff_theta = np.pi/3600

        current_line = lines[0]
        for i, line in enumerate(lines):
            rhoc, thetac = current_line[0]
            rho,theta = line[0]
            if(abs(rhoc - rho) > max_diff_rho) \
              and abs(rhoc - rho) > max_diff_theta:
                lines_combined.append(current_line)
                current_line = line
            else:
                current_line = [[(rhoc + rho) / 2.0, (thetac + theta) / 2.0]]
        lines_combined.append(current_line)

        lines =  lines_combined

        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            # 3000は画像の幅より大きければよい適当な数字
            x1 = int(x0 + 3000*(-b))
            y1 = int(y0 + 3000*(a))
            x2 = int(x0 - 3000*(-b))
            y2 = int(y0 - 3000*(a))
            cv2.line(img, (x1,y1),(x2,y2), (255,255,0), 2)

        # 交点を求める
        for i in range(0, len(lines)-1):
            for j in range(i + 1, len(lines)):
                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                a_i = np.cos(theta_i)
                b_i = np.sin(theta_i)
                a_j = np.cos(theta_j)
                b_j = np.sin(theta_j)

                # 平行なものは交わらない（計算すると除算例外が発生する）
                if theta_i == theta_j:
                    continue

                # p = x * consθ + x * sinθ* y
                # p' = x * consθ' + x * sinθ'* y
                # 連立方程式を解くと以下の式がもとまる
                x = ((rho_i * b_j) - (rho_j) * (b_i)) / ((a_i * b_j) - (a_j * b_i))
                y = ((rho_i * a_j) - (rho_j) * (a_i)) / ((a_j * b_i) - (a_i * b_j))

                # 求めた交点を中心に円を描く
                cv2.circle(img, (int(x),int(y)), 50, (0, 0, 255), 3)

        cv2.imshow('hough lines & cross points', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # # 台の領域を囲む矩形を取得する（長方形の回転が許されているが、台形はだめ）
        # rect = cv2.minAreaRect(pool)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # im = cv2.drawContours(img, [box], 0, (0,0,255), 2)

        # # 台形補正
        # # http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
        # peri = cv2.arcLength(pool, True)
        # approx = cv2.approxPolyDP(pool, 0.02*peri, True)
        # cv2.drawContours(img, [approx], -1, (0,255,0), 3)

        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # rect = cv2.minAreaRect(contours[2])
        # r = cv2.cv.BoxPoints(rect)


if __name__ == '__main__':
    main()
