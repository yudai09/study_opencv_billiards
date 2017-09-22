# -*- coding: utf-8

import cv2
import numpy as np
import pylab as plt


def main():
    images=[
        # 'photos/20170922/2017-09-22-170036.jpg',
        # 'photos/20170922/2017-09-22-170041.jpg',
        'photos/20170922/2017-09-22-170045.jpg',
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

        # HSVカラー空間に変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 台だけを抽出するために色のマスクを作成する
        # ref. http://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
        greenLower = (40, 60, 80)
        greenUpper = (90, 255, 255)

	mask = cv2.inRange(img, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

        # ボールと台の輪郭がちょっとギザるのでガウシアンフィルタで平滑化
        # あとでボールをハフ変換で円として検出するときに役に立つ
        mask = cv2.GaussianBlur(mask, (9,9), 1)

        # 輪郭検出
	_, contours, _  = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		                          cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭を面積が大きい順に並べ替える。
        contours.sort(key=cv2.contourArea, reverse=True)

        cv2.drawContours(img, contours, 0, (255,0,0), 3)

        # 面積が一番大きい物を台とみなす
        pool = contours[0]

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
        crosspoints = []
        print('lines')
        print(lines)

        for i in range(0, len(lines)-1):
            for j in range(i + 1, len(lines)):
                print('i,j')
                print([i,j])
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

                print('crosspoint')
                print([x,y])

                if abs(x) > 3000 or abs(y) > 3000:
                    print 'out of scope'
                    continue

                # 保管
                crosspoints.append([x,y])

        if(len(crosspoints) != 4):
            # ４つの角が取得できなかった場合（線がうまく検出できなかった場合など）
            print('cross points is not 4.')
            print(len(crosspoints))
            continue

        # あとから処理しやすいように画像左から台が長方形に表示されるように投影する
        # 左上、右上、右下、左下の順番にrectに入れる
        # ref. http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
        ###### 投影処理はじまり
        rect = np.zeros((4, 2), dtype = "float32")

        sum = np.sum(crosspoints, axis=1)
        rect[0] = crosspoints[np.argmin(sum)]
        rect[2] = crosspoints[np.argmax(sum)]

        diff = np.diff(crosspoints, axis=1)
        rect[1] = crosspoints[np.argmin(diff)]
        rect[3] = crosspoints[np.argmax(diff)]

        for x, y in rect:
            # 求めた交点を中心に円を描く
            cv2.circle(img, (int(x),int(y)), 50, (0, 0, 255), 3)

        cv2.imshow('hough lines & cross points', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # それぞれの直線の距離をもとめる（width, heightそれぞれ）
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # width, heightともに大きいほうを利用する
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        img_warped = cv2.warpPerspective(img_org, M, (maxWidth, maxHeight))
        mask_warped = cv2.warpPerspective(mask, M, (maxWidth, maxHeight))

        ###### 投影処理終わり

        # ここから投射したものだけ扱う
        img = img_warped
        img_org = img.copy()
        mask = mask_warped

        cv2.imshow('warp', img_org)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # BGR->HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 球の大きさはあくまで入力の画像に合わせているのにすぎないので、別の画像を使用する場合は要調整
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT,
                                   # param1の値は大きい方がよいっぽい
                                   dp=2, minDist=20,
                                   param1=200, param2=40, minRadius=26, maxRadius=30)
                                   # dp=2, minDist=5,

        if circles is None or len(circles) <= 0:
            raise Exception('cannot find any balls')

        # Work in progress
        # 球の番号を識別するために以下を実施する。
        # 球の色のヒストグラムを調べる。
        # ストライプがあるかどうかを全体に占める白色の割合から調べる
        hist_list = []
        for i, circle in enumerate(circles[0]):
            x, y, r = circle
            img_ball = img[int(y-r):int(y+r), int(x-r):int(x+r)]
            img_org_ball = img_org[int(y-r):int(y+r), int(x-r):int(x+r)]


            whiteLower = (130, 130, 130)
            whiteUpper = (255, 255, 255)

            # 白はRGBのほうが判定しやすい気がする
	    white = cv2.inRange(img_org_ball, whiteLower, whiteUpper)
	    # white = cv2.inRange(img_ball, greenLower, greenUpper)
	    white = cv2.erode(white, None, iterations=2)
	    white = cv2.dilate(white, None, iterations=2)

            # 白色の範囲を数値化
            print('np.sum(white)')
            print(np.sum(white))

            cv2.imshow('img_ball', img_org_ball)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('white', white)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            plt.subplot(len(circles[0]),3,3*i+1)
            plt.imshow(img_org_ball)
            plt.subplot(len(circles[0]),3,3*i+2)
            plt.imshow(img_ball)

            hist = cv2.calcHist([img_ball],[0],None,[256],[0,256])
            hist_list.append(hist)
            plt.subplot(len(circles[0]),3,3*i+3)
            plt.plot(hist)
            plt.xlim([0,256])

            # ボールの外接円を描画
            cv2.circle(img, (x, y), r, (255, 255, 0), 2)
            # ↑を囲う四角を描画
            cv2.rectangle(img, (x-r,y-r), (x+r, y+r), (255,255,0),1)

        # plt.ion()
        plt.show()

        cv2.imshow('pool', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(hist_list)

        scores = np.zeros((len(hist_list), len(hist_list)))
        for i in range(0, len(hist_list)-1):
            for j in range(i+1, len(hist_list)):
                hist_i = hist_list[i]
                hist_j = hist_list[j]

                score = cv2.compareHist(hist_i, hist_j, cv2.HISTCMP_CORREL)
                if score >= 0.8:
                    scores[i,j] = 1
                print('hists')
                print([hist_i,hist_j])
                print('score')
                print(score)

        print scores

if __name__ == '__main__':
    main()
