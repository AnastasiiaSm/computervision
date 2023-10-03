import cv2


video = cv2.VideoCapture(0) #видео с веб-камеры

ret, frame1 = video.read()
ret, frame2 = video.read() #создаём два изображения

while video.isOpened():

    difference = cv2.absdiff(frame1, frame2) #абсолютная разница между двумя изображениями

    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY) #переводим изображение в чёрно-белые цвета

    blur = cv2.GaussianBlur(gray, (11,11), 0) #размываем изображение, чтобы убрать шум

    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) #устанавливаем пороговое значение

    dilate = cv2.dilate(threshold, None, iterations=3) #расширяем изображение

    contour, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #ищем контуры

    cv2.drawContours(frame1, contour, -1, (255, 0, 139), 2) #рисуем контуры

    cv2.imshow("image", frame1) #выводим изображение

    frame1 = frame2
    ret, frame2 = video.read()

    if cv2.waitKey(5) == 27:
        break                 #выход на esc

video.release()