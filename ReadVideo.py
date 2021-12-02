import cv2 as cv
cap=cv.VideoCapture('medium.avi')
count=0
while 1:
    ret, frame = cap.read()

    #frame=cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    #frame=cv.resize(frame,(320,240))
    cv.imwrite("./medium/"+ str(count)+".jpg",frame)
    count=count+1
    cv.imshow("a",frame)
    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
 
cap.release()
cv.destroyAllWindows()