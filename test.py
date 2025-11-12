

# cap.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import cvzone
import math
import PokerHandFunction

# Load the YOLO model
model = YOLO("playingCards.pt")

classNames = ['10C', '10D', '10H', '10S', 
              '2C', '2D', '2H', '2S', 
              '3C', '3D', '3H', '3S', 
              '4C', '4D', '4H', '4S', 
              '5C', '5D', '5H', '5S', 
              '6C', '6D', '6H', '6S', 
              '7C', '7D', '7H', '7S', 
              '8C', '8D', '8H', '8S', 
              '9C', '9D', '9H', '9S', 
              'AC', 'AD', 'AH', 'AS', 
              'JC', 'JD', 'JH', 'JS', 
              'KC', 'KD', 'KH', 'KS', 
              'QC', 'QD', 'QH', 'QS']

# Video file paths
input_video_path = "16085855.mp4"
output_video_path = "output_1.mp4"

cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    results = model(img, stream=True)
    hand = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)

            if conf > 0.5:
                hand.append(classNames[cls])

    hand = list(set(hand))  # Remove duplicates

    if len(hand) == 5:
        results_hand = PokerHandFunction.findPokerHand(hand)
        cvzone.putTextRect(img, f'Results: {results_hand}', (150, 30), scale=1, thickness=1)

    # Write the frame to the output video
    out.write(img)

    # Optional: Show live video
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved at: {output_video_path}")
