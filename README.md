# 스마트 출석 카메라 (FaceDetectionProject)
- 201644039 강동찬
- 201644064 임태현
- 201644092 장태영
- 201344056 윤종섭

# 목차
+ 프로젝트 목적
+ 프로젝트 목표 와 기대효과
+ 프로젝트 주요내용
	- 사용프로그램 및 모듈
	- 구현방법 
	- 프로그램 코드 및 설명
	- 프로그램 동작원리요약
+ 프로젝트 결과물 


# 프로젝트 목적
+ 라즈베리파이 및 OpenCV 기반으로 하는 software및 hardware 개발 프로젝트로서 
각 사용자들의 얼굴 인식을 하여 사용자들 사이에서 이름, 유사도 정보를 공유할 수 있게 의사 소통하는 시스템을 구현합니다

# 프로젝트 목표 및 기대효과
### 목표
+ 화상회의 참여자들의 얼굴을 촬영. 이를 학습시켜 각 참여자들의 신분을 나타낸다.

### 기대효과
+ 출석에 활용하는 시간 감소 
+ 실시간 촬영을 통한 회의 참여 유도 및 효율향상
+ 원격회의 서비스 활용도 증가

# 프로젝트 주요내용 
## 1. 사용프로그램 및 모듈
### 1) OpenCv
![opencv](https://user-images.githubusercontent.com/71091406/101994107-34810900-3d03-11eb-9bd7-680510086ecf.png)

1. OpenSource Computer Vision의 약자로 
    다양한 영상/동영상 처리에 사용할 수 있는 오픈소스 라이브러리 

2. C++, C, Python 및 Java와 같은 다양한 인터페이스를 지원
    Windows, Linux, Mac OS, iOS 및 Android같은 다양한 OS를 지원

3. OpenCV는 멀티 코어 프로세싱을 지원하기 때문에 다양한 상황에 응용이 가능
   (예를 들어 윤곽선 검출,  노이즈 제거,  이미지 스티칭을 이용한 파노라믹 사진제작)
   
4. 실시간 이미지 프로세싱에 중점을 둔 라이브러리


### 2) Haar Cascade

1. 머신 러닝기반의 오브젝트 검출 알고리즘

2. 비디오 또는 이미지에서 오브젝트를 검출하기 위해 사용, 직사각형 영역으로 구성되는 특징을 사용기 때문에 픽셀을 직접 사용할 때 보다 동작 속도가 빠름

3. 사람의 얼굴을 인식하기위한 Haar Cascade 방식의 알고리즘은 머신 러닝의 컨볼루션 신경망 분석 기법과 유사

4. Haarcascade 라이브러리를 사용하여 인식할 수 있는 오브젝트는 사람의 정면 얼굴, 얼굴 안의 눈, 고양이 얼굴, 사람의 몸 각 부분들, 컬러 및 차량을 포함

#### Haar Cascade 알고리즘 4단계

1. Haar Feature Selection 
    -  이미지 전체 스캔하고  하르특징 계산하여 영역내 픽셀 합의 차이 이용
2. Creating  Integral Images 
    - 픽셀의 합을 구하기는 것을 빠르게 하기 위해 적분 이미지 사용
3. Adaboost Training
   - 얼굴 검출을 하는데 도움이 되는 의미 있는 특징계산
4. Cascading Classifiers 
   -  현재 윈도우가 있는 영역이 얼굴 영역인지를 단계별로  체크

## 2. 구현방법
![구현방법](https://user-images.githubusercontent.com/71091406/101994156-74e08700-3d03-11eb-82a2-92474e30cbe0.png)

## 3. 프로그램 코드 및 설명

### 1) face_detection.py
**OpenCV를 사용해 영상을 촬영하고 영상에 촬영된 사람의 얼굴을 인식한다.
**얼굴인식은 Haar feature-based cascade classifiers를 사용한다.
```
import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(3,640) 
cap.set(4,480) 
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video',img) 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
cap.release()
cv2.destroyAllWindows()
```

### 2) face_dataset.py
```
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


#for each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

#Initialize individual sampling face count
count = 0
while(True):
	ret, img = cam.read()
	#img = cv2.flip(img, -1) #flip video image vertically
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_detector.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		count += 1
		# Save the captured image into the datasets folder
		cv2.imwrite("dataset/user." + str(face_id) + '.' + str(count) + ".jpg",gray[y:y+h,x:x+w])
		cv2.imshow('image', img)
	k = cv2.waitKey(100) & 0xff #  press 'ESC' for exiting video
	if k == 27:
		break
	elif count >= 30: # Take 30 face sample and stop video
		break

#Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
```

### 3) face_training.py
```
import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
```
### 4) 판별: face_recognition.py
**OpenCV 라이브러리를 사용하여 실시간으로 인식된 얼굴과 trainer.yml을 비교하여 이름과 유사도를 측정한다.**
```
import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> loze: id=1,  etc
# 사용자 이름 설정
names = ['chan', 'LimTaeHyun', 'JangTaeYoung', 'Jongsub']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
```
## 4. 프로그램 동작원리 요약

1. OpenCV의 haar cascades를 이용하여  각 사용자의 얼굴을 인식하여 흑백 이미지 파일로 저장

2. 미리 저장해 두었던 각 사용자별 얼굴 이미지를 통하여 이를 OpenCV의 라이브러리를 통해 학습시키고, 이를 yml 파일로 저장

3. 인식된 얼굴과 trainer.yml 파일에 추출된 각 사용자의 특징과 일치하는지 확인해서 알려준다.

# 프로젝트 결과물
### 사진 인식
[![Video Label](https://user-images.githubusercontent.com/54888988/101995574-3e5c3980-3d0e-11eb-8f1f-70c54202fe0f.png)](https://www.youtube.com/watch?v=8FkADlbuME8?t=0s)
---
### 화상회의 얼굴인식(zoom) 
[![Video Label](https://img.youtube.com/vi/U9Vv9ufDmBs/0.jpg)](www.youtube.com/watch?v=U9Vv9ufDmBs?t=0s)
