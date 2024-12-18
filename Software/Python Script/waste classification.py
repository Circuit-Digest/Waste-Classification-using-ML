import cv2
from inference_sdk import InferenceHTTPClient
import serial
import time  # Import time module to add delays

# Configure the serial connection
ser = serial.Serial('/dev/ttyACM0', baudrate=115200)
print("Press Enter to capture a frame or 'q' to exit the live camera feed.")

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="iEtMvLxXrAR7cF9eFdTV"
)

last_label = ""  # To store the last label value that was sent to serial
inference_in_progress = False  # Flag to track if inference is in progress

try:
    while True:
        # Read the current frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture an image from the webcam.")
            continue

        # Show the live camera feed
        cv2.imshow("Live Camera Feed", frame)

        # Check for keyboard input for capturing a frame
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press
        if key == ord('q'):  # Press 'q' to quit
            break
        elif key == 13 and not inference_in_progress:  # Press Enter to capture a frame (if no inference is in progress)
            print("Capturing frame and sending for inference...")

            # Set the flag to indicate that inference is in progress
            inference_in_progress = True

            # Send the current frame for inference
            result = CLIENT.infer(frame, model_id="garbage-classification-3/2")

            # Parse the inference result
            predictions = result.get('predictions', [])

            if predictions:
                # Find the prediction with the highest confidence
                best_prediction = max(predictions, key=lambda pred: pred['confidence'])
                x = int(best_prediction['x'])
                y = int(best_prediction['y'])
                width = int(best_prediction['width'])
                height = int(best_prediction['height'])
                label = best_prediction['class']
                confidence = best_prediction['confidence']

                # Calculate top-left and bottom-right coordinates
                top_left = (x - width // 2, y - height // 2)
                bottom_right = (x + width // 2, y + height // 2)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(frame, label_text, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Display the result in the console
                print(f"Detected: {label}, Confidence: {confidence:.2f}, Bounding Box: [{x}, {y}, {width}, {height}]")

                # Send the label value to the serial port if it's not the same as the last sent
                if label != last_label:
                    last_label = label
                    ser.write(f"{label}\n".encode())
            else:
                print("No predictions made.")

            # Add a short delay to avoid sending too many requests
            time.sleep(1)  # 1 second delay before allowing the next inference

            # Reset the flag after the inference is done
            inference_in_progress = False

except KeyboardInterrupt:
    print("\nExiting...")

# Release the resources
cap.release()
cv2.destroyAllWindows()
ser.close()
