import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

Part = int(input("專案選擇(1.圖像變形、2.小波轉換、3.霍夫轉換):"))

if Part == 1:  #圖像變形
    image = cv2.imread('HW6/C1HW06-2022/Part 1 Image',cv2.IMREAD_GRAYSCALE) #加載灰階圖
    
    #梯形變形
    k = float(input("左右調整(0.1-0.3): "))
    h = float(input("上下調整(1.0-1.5): "))
    height, width = image.shape[0], image.shape[1]
    value = k*height
    resize_gray = np.zeros([int(height*h), width], np.uint8)
    for i in range (height) :
        temp = int( value + k * i ) 
        for j in range (temp,width-temp) :
            #每行非黑色區域的長度
            distance = int(width-temp) - int(temp-5)
            #缩小倍率
            ratio = distance / width
            #取點距離
            stepsize = 1.0/ratio
            #將同意行缩小相同倍率
            resize_gray[i][j] = image[i][int((j-temp)*stepsize)]


    #垂直+水平方向變形
    rows,cols = image.shape
    output = np.zeros(image.shape,dtype = image.dtype)   
    for i2 in range(rows):
       for j2 in range(cols):
         offset_x = int(50.0*math.cos(2*math.pi*i2/180)) #x=u+振幅*sin(2.0*π*v/頻率)
         offset_y = int(50.0*math.sin(2*math.pi*j2/180)) #y=v+振幅*sin(2.0*π*u/頻率)
         if j2+offset_x < cols and i2+offset_y < rows:
            output[i2,j2] = image[(i2+offset_y)%rows,(j2+offset_x)%cols]
         else:
            output[i2,j2] = 255
            
    #圓形變形
    img = cv2.imread('IP_dog.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heights, widths = img.shape[:2]
    heights = int(heights)
    widths = int(widths)
    #生成內顯示模板
    circleIn = np.zeros((heights, widths, 1), np.uint8)
    circleIn = cv2.circle(circleIn, (width // 2, height // 2), min(height, width) // 2, (1), -1)
    #生成外顯示模板
    circleOut = circleIn.copy()
    circleOut[circleOut == 0] = 2
    circleOut[circleOut == 1] = 0
    circleOut[circleOut == 2] = 1
    #原圖與内顯示模板融合
    #生成空白陣列
    imgIn = np.zeros((height, width, 4), np.uint8)
    #複製前3個通道
    imgIn[:, :, 0] = np.multiply(img[:, :, 0], circleIn[:, :, 0])
    imgIn[:, :, 1] = np.multiply(img[:, :, 1], circleIn[:, :, 0])
    imgIn[:, :, 2] = np.multiply(img[:, :, 2], circleIn[:, :, 0])
    #設置α通道的不透明部分
    circleIn[circleIn == 1] = 255
    imgIn[:, :, 3] = circleIn[:, :, 0]
    imgIn = cv2.cvtColor(imgIn, cv2.COLOR_RGB2GRAY)

    plt.figure(figsize=(10,5)) 
    plt.suptitle('Part 1') 
    plt.subplot(2,2,1), plt.title('Image')
    plt.imshow(image,cmap='gray'), plt.axis('off')
    plt.subplot(2,2,2), plt.title('Trapezoid')
    plt.imshow(resize_gray,cmap='gray'), plt.axis('off')
    plt.subplot(2,2,3), plt.title('Vertical & Horizontal Wave')
    plt.imshow(output,cmap='gray'), plt.axis('off')
    plt.subplot(2,2,4), plt.title('Circular')
    plt.imshow(imgIn,cmap='gray'), plt.axis('off')
    
elif Part == 2:  #小波轉換
    imgA = cv2.imread("multifocus1.JPG") #載入圖片A
    imgB = cv2.imread("multifocus2.JPG") #載入圖片B
    heigh, wide, channel = imgA.shape #获取图像的高、宽、通道数

    #臨時變數、存儲哈爾小波處理的數據#
    tempA1 = []                   #
    tempA2 = []                   #
    tempB1 = []                   #
    tempB2 = []                   #

    waveImgA = np.zeros((heigh, wide, channel), np.float32) #存储A圖片小波處理數據的變數
    waveImgB = np.zeros((heigh, wide, channel), np.float32) #存储B圖片小波處理數據的變數
    
    #水平方向的哈爾小波處理，對圖片的B、G、R三个通道分别進行
    for c in range(channel):
    	for x in range(heigh):
            for y in range(0,wide,2):
                tempA1.append((float(imgA[x,y,c]) + float(imgA[x,y+1,c]))/2) 
                tempA2.append((float(imgA[x,y,c]) + float(imgA[x,y+1,c]))/2 - float(imgA[x,y,c])) 
                tempB1.append((float(imgB[x,y,c]) + float(imgB[x,y+1,c]))/2) 
                tempB2.append((float(imgB[x,y,c]) + float(imgB[x,y+1,c]))/2 - float(imgB[x,y,c])) 
            tempA1 = tempA1 + tempA2 
            tempB1 = tempB1 + tempB2 
            for i in range(len(tempA1)):
                waveImgA[x,i,c] = tempA1[i] 
                waveImgB[x,i,c] = tempB1[i] 
            tempA1 = [] 
            tempA2 = [] 
            tempB1 = []
            tempB2 = []
 
    #垂直方向哈爾小波處理，與水平方向同理
    for c in range(channel):
        for y in range(wide):
            for x in range(0,heigh-1,2):
                tempA1.append((float(waveImgA[x,y,c]) + float(waveImgA[x+1,y,c]))/2)
                tempA2.append((float(waveImgA[x,y,c]) + float(waveImgA[x+1,y,c]))/2 - float(waveImgA[x,y,c]))
                tempB1.append((float(waveImgB[x,y,c]) + float(waveImgB[x+1,y,c]))/2)
                tempB2.append((float(waveImgB[x,y,c]) + float(waveImgB[x+1,y,c]))/2 - float(waveImgB[x,y,c]))
            tempA1 = tempA1 + tempA2
            tempB1 = tempB1 + tempB2
            for i in range(len(tempA1)):
                waveImgA[i,y,c] = tempA1[i]
                waveImgB[i,y,c] = tempB1[i]
            tempA1 = []
            tempA2 = []
            tempB1 = []
            tempB2 = []
 
    #求以x,y为中心的5x5矩陣的方差
    varImgA = np.zeros((heigh//2, wide//2, channel), np.float32) #將圖像A中低频數據求方差後存储的變數
    varImgB = np.zeros((heigh//2, wide//2, channel), np.float32) #將像B中低频數據求方差後存储的變數
    for c in range(channel):
        for x in range(heigh//2):
            for y in range(wide//2):
			#############################
			#對圖片邊界的像素點進行處理
              if x - 3 < 0:
                  up = 0
              else:
                  up = x - 3
              if x + 3 > heigh//2:
                  down = heigh//2
              else:
                  down = x + 3
              if y - 3 < 0:
                  left = 0
              else:
                  left = y - 3
              if y + 3 > wide//2:
                  right = wide//2
              else:
                  right = y + 3
			
            meanA, varA = cv2.meanStdDev(waveImgA[up:down,left:right,c]) #求圖片A以x,y為中心的5x5矩陣的方差，mean表示平均值，var表示方差
            meanB, varB = cv2.meanStdDev(waveImgB[up:down,left:right,c]) #求圖片B以x,y為中心的5x5矩陣的方差
 
            varImgA[x,y,c] = varA #將圖片A對應位置像素的方差存储在變數中
            varImgB[x,y,c] = varB #將圖片B對應位置像素的方差存储在變數中
            
            #求兩圖的權重
    weightImgA = np.zeros((heigh//2, wide//2, channel), np.float32) 
    weightImgB = np.zeros((heigh//2, wide//2, channel), np.float32) 
    for c in range(channel):
        for x in range(heigh//2):
            for y in range(wide//2):
                weightImgA[x,y,c] = varImgA[x,y,c] / (varImgA[x,y,c]+varImgB[x,y,c]+0.00000001) 
                weightImgB[x,y,c] = varImgB[x,y,c] / (varImgA[x,y,c]+varImgB[x,y,c]+0.00000001) 
 
    #進行融合，高頻————係数绝对值最大化，低頻————局部方差準則
    reImgA = np.zeros((heigh, wide, channel), np.float32) 
    reImgB = np.zeros((heigh, wide, channel), np.float32)
    for c in range(channel):
        for x in range(heigh):
            for y in range(wide):
                if x < heigh//2 and y < wide//2:
                    reImgA[x,y,c] = weightImgA[x,y,c]*waveImgA[x,y,c] + weightImgB[x,y,c]*waveImgB[x,y,c] #對兩張圖片低頻的地方進行權值融合數據
                else:
                    reImgA[x,y,c] = waveImgA[x,y,c] if abs(waveImgA[x,y,c]) >= abs(waveImgB[x,y,c]) else waveImgB[x,y,c] #對兩張高頻的進行绝對值係数最大規則融合
 
    #垂直方向進行重購
    for c in range(channel):
        for y in range(wide):
            for x in range(heigh):
                if x%2 == 0:
                    reImgB[x,y,c] = reImgA[x//2,y,c] - reImgA[x//2 + heigh//2,y,c] #根據哈爾小波原理，將重購的數據存儲在臨時變數中
                else:
                    reImgB[x,y,c] = reImgA[x//2,y,c] + reImgA[x//2 + heigh//2,y,c] #圖片的前半段是低頻，後半段是高頻
 
    #水平方向進行重構，與垂直方向同理
    for c in range(channel):
        for x in range(heigh):
            for y in range(wide):
                if y%2 ==0:
                    reImgA[x,y,c] = reImgB[x,y//2,c] - reImgB[x,y//2 + wide//2,c]
                else:
                    reImgA[x,y,c] = reImgB[x,y//2,c] + reImgB[x,y//2 + wide//2,c]
    #限制圖像的範圍(0-255),若不限制，根據np.astype(np.uint8)的規則，會對圖片產生噪聲
    reImgA[reImgA[:, :, :] < 0] = 0
    reImgA[reImgA[:, :, :] > 255] = 255

    cv2.imwrite('multifocusO.JPG', reImgA.astype(np.uint8))

#------------------------------------------------------------------------------
    #載入新生成的圖片，進行三張圖片的小波轉換
    imgC = cv2.imread("multifocusO.JPG") 
    imgD = cv2.imread("multifocus3.JPG") 
    heigh1, wide1, channel1 = imgC.shape 

    tempC1 = []                   
    tempC2 = []                   
    tempD1 = []                   
    tempD2 = []                   

    waveImgC = np.zeros((heigh1, wide1, channel1), np.float32) 
    waveImgD = np.zeros((heigh1, wide1, channel1), np.float32) 

    for c1 in range(channel1):
        for x1 in range(heigh1):
            for y1 in range(0,wide1,2):
                tempC1.append((float(imgC[x1,y1,c1]) + float(imgC[x1,y1+1,c1]))/2) 
                tempC2.append((float(imgC[x1,y1,c1]) + float(imgC[x1,y1+1,c1]))/2 - float(imgC[x1,y1,c1])) 
                tempD1.append((float(imgD[x1,y1,c1]) + float(imgD[x1,y1+1,c1]))/2) 
                tempD2.append((float(imgD[x1,y1,c1]) + float(imgD[x1,y1+1,c1]))/2 - float(imgD[x1,y1,c1]))
            tempC1 = tempC1 + tempC2 
            tempD1 = tempD1 + tempD2 
            for i1 in range(len(tempC1)):
                waveImgC[x1,i1,c1] = tempC1[i1] 
                waveImgD[x1,i1,c1] = tempD1[i1] 
            tempC1 = [] 
            tempC2 = [] 
            tempD1 = []
            tempD2 = []
 
    for c1 in range(channel1):
       for y1 in range(wide1):
           for x1 in range(0,heigh1-1,2):
               tempC1.append((float(waveImgC[x1,y1,c1]) + float(waveImgC[x1+1,y1,c1]))/2)
               tempC2.append((float(waveImgC[x1,y1,c1]) + float(waveImgC[x1+1,y1,c1]))/2 - float(waveImgC[x1,y1,c1]))
               tempD1.append((float(waveImgD[x1,y1,c1]) + float(waveImgD[x1+1,y1,c1]))/2)
               tempD2.append((float(waveImgD[x1,y1,c1]) + float(waveImgD[x1+1,y1,c1]))/2 - float(waveImgD[x1,y1,c1]))
           tempC1 = tempC1 + tempC2
           tempD1 = tempD1 + tempD2
           for i1 in range(len(tempC1)):
               waveImgC[i1,y1,c1] = tempC1[i1]
               waveImgD[i1,y1,c1] = tempD1[i1]
           tempC1 = []
           tempC2 = []
           tempD1 = []
           tempD2 = []
    varImgC = np.zeros((heigh1//2, wide1//2, channel1), np.float32) 
    varImgD = np.zeros((heigh1//2, wide1//2, channel1), np.float32) 
    for c1 in range(channel1):
        for x1 in range(heigh1//2):
            for y1 in range(wide1//2):
                if x1 - 3 < 0:
                    up1 = 0
                else:
                    up1 = x1 - 3
                if x1 + 3 > heigh1//2:
                    down1 = heigh1//2
                else:
                    down1 = x1 + 3
                if y1 - 3 < 0:
                    left1 = 0
                else:
                    left1 = y1 - 3
                if y + 3 > wide1//2:
                    right1 = wide1//2
                else:
                    right1 = y1 + 3
			
                meanC, varC = cv2.meanStdDev(waveImgC[up1:down1,left1:right1,c1])
                meanD, varD = cv2.meanStdDev(waveImgD[up1:down1,left1:right1,c1])
                
                varImgC[x1,y1,c1] = varC 
                varImgD[x1,y1,c1] = varD 
 
    
    weightImgC = np.zeros((heigh1//2, wide1//2, channel1), np.float32) 
    weightImgD = np.zeros((heigh1//2, wide1//2, channel1), np.float32) 
    for c1 in range(channel1):
        for x1 in range(heigh1//2):
            for y1 in range(wide1//2):
                weightImgC[x1,y1,c1] = varImgC[x1,y1,c1] / (varImgC[x1,y1,c1]+varImgD[x1,y1,c1]+0.00000001) 
                weightImgD[x1,y1,c1] = varImgD[x1,y1,c1] / (varImgC[x1,y1,c1]+varImgD[x1,y1,c1]+0.00000001) 
 
    reImgC = np.zeros((heigh1, wide1, channel1), np.float32) 
    reImgD = np.zeros((heigh1, wide1, channel1), np.float32) 
    for c1 in range(channel1):
        for x1 in range(heigh1):
            for y1 in range(wide1):
                if x1 < heigh1//2 and y1 < wide1//2:
                    reImgC[x1,y1,c1] = weightImgC[x1,y1,c1]*waveImgC[x1,y1,c1] + weightImgD[x1,y1,c1]*waveImgD[x1,y1,c1] 
                else:
                    reImgC[x1,y1,c1] = waveImgC[x1,y1,c1] if abs(waveImgC[x1,y1,c1]) >= abs(waveImgD[x1,y1,c1]) else waveImgD[x1,y1,c1] 
 
    for c1 in range(channel1):
        for y1 in range(wide1):
            for x1 in range(heigh1):
                if x1%2 == 0:
                    reImgD[x1,y1,c1] = reImgC[x1//2,y1,c1] - reImgC[x1//2 + heigh1//2,y1,c1] 
                else:
                    reImgD[x1,y1,c1] = reImgC[x1//2,y1,c1] + reImgC[x1//2 + heigh1//2,y1,c1] 
 

    for c1 in range(channel1):
        for x1 in range(heigh1):
            for y1 in range(wide1):
                if y1%2 ==0:
                    reImgC[x1,y1,c1] = reImgD[x1,y1//2,c1] - reImgD[x1,y1//2 + wide1//2,c1]
                else:
                    reImgC[x1,y1,c1] = reImgD[x1,y1//2,c1] + reImgD[x1,y1//2 + wide1//2,c1]

    reImgC[reImgC[:, :, :] < 0] = 0
    reImgC[reImgC[:, :, :] > 255] = 255
    cv2.imwrite('multifocus.JPG', reImgC.astype(np.uint8))
    cv2.imshow("multifocus1", imgA)
    cv2.imshow("multifocus2", imgB)
    cv2.imshow("multifocus3", imgD)
    cv2.imshow("multifocus", reImgC.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif Part == 3:  #霍夫轉換
    #輪廓邊線相交
    def line_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray,None,fx=1.55, fy=1.55, interpolation = cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)  #apertureSize參數默認其實就是3
        #cv2.imshow("edges", edges)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 118)
        result = gray.copy()
        for line in lines:
            rho = line[0][0]  #第一個元素是距離rho
            theta= line[0][1] #第二個元素是角度theta
            if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直線
                pt1 = (int(rho/np.cos(theta)),0)               #該直線與第一行的交點
                #该直線與最後一行的焦點
                pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
                cv2.line( result, pt1, pt2, (0,0,0), (5))            
  
            else:                                                  #水平直線
                pt1 = (0,int(rho/np.sin(theta)))               # 該直線與第一列的交點
            pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
            cv2.line(result, pt1, pt2, (0,0,0), (5))           
            cv2.imshow("image-lines", result)

        #統計概率霍夫線變換
    def line_detect_possible_demo(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize參數默認其實就是3
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=120, maxLineGap=5)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow("line_detect_possible_demo",image)
    
        #面積計算
    def Area_perimeter(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray,(3,3),0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        ret , binary = cv2.threshold (edges , 127 , 255 , cv2.THRESH_BINARY)
        _,contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt,True) #周長計算
        area = cv2.contourArea(cnt) #面積計算
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        image_1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.polylines(image_1, [approx], True, (0, 0, 255), 2)
        imgnew1 = cv2.drawContours(image_1, approx, -1, (0,0,255), 3)
        print("周長: "+str(perimeter))
        print("面積: "+str(area))
        cv2.imshow("Area_perimeter",imgnew1)

    src = cv2.imread("rects.bmp")
    cv2.imshow("input_image", src)
    line_detection(src)
    line_detect_possible_demo(src)
    Area_perimeter(src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
       