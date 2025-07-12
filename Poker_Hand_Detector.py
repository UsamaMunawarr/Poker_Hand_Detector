from ultralytics import YOLO # Import the YOLO class from the ultralytics package
import cv2 # Import OpenCV for image processing
import cvzone # Import cvzone for additional computer vision functionalities
import math # Import math for mathematical operations
import PokerHandFunction
#######################################
### Webcam Setup and Model Loading ###
#######################################
cap = cv2.VideoCapture(2)  # For webcam
cap.set(3, 720)  # Set the width of the webcam feed
cap.set(4, 480)   # Set the height of the webcam feed


######################################
## For Videos Use this code instead ##
########################################
# cap = cv2.VideoCapture(r"C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\Videos\people.mp4")# For video file

model = YOLO(r'C:\Users\usama\OneDrive\Desktop\Compurt_Vision_Proj\06_poker\Poker_Hand_Detector\playingCards.pt')  # Load the YOLOv8 model weights

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




while True:
    success, img = cap.read()  # Read a frame from the webcam
    results = model(img, stream=True)  # Run inference on the webcam feed
    hand = []  # Initialize an empty list to store the detected cards

    for r in results:
        boxes = r.boxes  # Get the bounding boxes from the results
        for box in boxes:
            ##########################################
            # Get the coordinates of the bounding box
            ##########################################
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            w, h = x2-x1,y2-y1  # Calculate width and height from coordinates

            #############################################
            # if we press control and right click on cornerRect blow chunk it will show the code of 
            # cornerRect and we can change the corner of rectangles accordingly 
            ##############################################
            cvzone.cornerRect(img, (x1, y1, w, h)) # Draw a rectangle around the detected object

            ################################################
            #### Display Class Name and Confidence Score ###
            ##################################################
            conf = math.ceil((box.conf[0]*100))/100  # Get the confidence score of the detection
            # print(f'Confidence: {conf}')  # Print the confidence score
            # cvzone.putTextRect(img, f'Conf: {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2) # Display the confidence score on the image

            ###########################################
            #### Class Name Display on Image ######
            ##########################################
            cls = int(box.cls[0]) # Get the className Variable id suppose 0 == person, 1 == bicycle, etc.
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)  # Display the class name and confidence score on the image

            if conf > 0.5:  # Only consider detections with confidence above 0.5
                hand.append(classNames[cls])  # Append the detected card to the hand list
        



    print(hand)
    hand = list(set(hand))  # Remove duplicates from the hand list
    print(hand)

    if len(hand) == 5:  # If we have detected 5 cards, we can process the hand
        results = PokerHandFunction.findPokerHand(hand)  # Call the function to find the poker hand
        print(f"Detected Poker Hand: {results}")  # Print the detected poker hand
        cvzone.putTextRect(img, f'Results: {results}', (150, 30), scale=1, thickness=1)  # Display the detected poker hand on the image

    cv2.imshow("Image", img)  # Display the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for a key press for 1 millisecond
        break








##############################################
#### Code to chekc which webcam i am using####
##############################################
# for i in range(3):  # Try indexes 0, 1, and 2 manually
#     print(f"Trying camera index {i}")
#     cap = cv2.VideoCapture(i)
#     if not cap.isOpened():
#         print(f"Camera {i} not available")
#         continue

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Failed to grab frame from camera {i}")
#             break

#         cv2.imshow(f"Camera {i}", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
