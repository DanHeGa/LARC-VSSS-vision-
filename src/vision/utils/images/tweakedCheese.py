import cv2
import os

cap = cv2.VideoCapture(2)


output_dir = "src/vision/vision/outputs4"
os.makedirs(output_dir, exist_ok=True)
count = 0

while True:
    success, frame = cap.read() 
    if not success:
        print("Failed to capture image")
        break
    # cut = cv2.resize(frame, (640, 480))
    cv2.imshow("Webcam Feed", frame)
    key = cv2.waitKey(1) & 0xFF 
    print("shape" + str(frame.shape))

    if key == ord('q'):
        break
    if key == ord('c'):
        filename = os.path.join(output_dir, f"captura_{count}.png")
        cv2.imwrite(filename, cut)
        print(f"captura: {filename}")
        count += 1


cap.release()
cv2.destroyAllWindows()

