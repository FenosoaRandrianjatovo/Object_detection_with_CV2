import cv2

def main(cam=False, 
        file="video",
        model='haarcascade_frontalface_default.xml') -> None:

    captured=cv2.VideoCapture(0) if cam else cv2.VideoCapture(f"{file}"+".mp4")
    
    print(f"the Video is well imported ......")
    "initializing the face classifier"
    face=cv2.CascadeClassifier(model)

    while True:
        _,image=captured.read()
        #now i convert the image into grayscale
        image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #detect all faces in the image
        faces=face.detectMultiScale(image_gray,1.3,5)
        #for every face draw a rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x,y, width,height in faces:
            cv2.circle(image,(x+width//2,y+height//2),radius=30,color=(255,22,0),thickness=3)
            # cv2.rectangle(image,(x,y),(x+width,y+height),color=(255,22,0),thickness=3)
            # cv2.putText(image,'Tava',(10,500),font,1,(0,0,255),2)
            cv2.putText(image,'Oron\'olona',(x+width//5,y-30),font,2,(10,22,255),5)
        cv2.imshow('Stark',image)
        if cv2.waitKey(1)==ord('q') or cv2.waitKey(1)==ord('w') :
            break
    captured.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    #main(False, "video1")
    main(True)
        
