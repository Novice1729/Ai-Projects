import cv2
import os

user_name = input("Enter the user's name: ")
user_id = input("Enter the user's ID: ")

save_path = f'C:\\Users\\ADMIN\\Desktop\\sumago\\Face_Recognition\\User_Images\\{user_name}_{user_id}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('C:\\Users\\ADMIN\\Desktop\\sumago\\Face_Recognition\\Images\\haarcascade_frontalface_default.xml')
faces_data = []
i = 0
img_count = 0

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_image = cv2.resize(crop_img, (160, 160))  

        if len(faces_data) < 100 and i % 5 == 0:  
            img_count += 1
            img_filename = os.path.join(save_path, f'{user_name}_{user_id}_{img_count}.jpg')
            cv2.imwrite(img_filename, resized_image)
            faces_data.append(resized_image)

        i += 1
        cv2.putText(frame, f"Captured: {len(faces_data)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 50), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    # Exit when 'q' is pressed or when 30 images are captured
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

print(f"Images saved for {user_name} (ID: {user_id}) in the folder: {save_path}")
