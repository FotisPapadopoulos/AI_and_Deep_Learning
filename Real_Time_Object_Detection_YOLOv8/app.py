import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

def main():
    # 1. Initialize YOLOv8 Nano model (lightweight, runs in real-time on CPU)
    # The library will automatically download 'yolov8n.pt' on its first run
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    
    # 2. Setup directory to save screenshots for the report
    screenshot_dir = "screenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
        print(f"Created folder '{screenshot_dir}' to store your report figures.")

    # 3. Initialize camera (using default resolution which is safer)
    print("Initializing camera feed...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("\n" + "="*50)
    print("YOLOv8 Object Recognition Application Active!")
    print("Keyboard Controls:")
    print("  - Press [S] to capture a screenshot for your report")
    print("  - Press [Q] to quit the application")
    print("="*50 + "\n")

    screenshot_counter = 1

    cv2.namedWindow("AIDL_03 Object Recognition (YOLOv8)", cv2.WINDOW_NORMAL)
    
    while True:
        start_time = time.time()
        
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break
            
        # Keep a clean copy of the frame (raw image) in case we want to save it
        raw_frame = frame.copy()
        
        # 4. Run YOLOv8 inference on the frame
        # verbose=False prevents cluttering the terminal output with raw detection lists
        results = model(frame, verbose=False)[0]
        
        # Extract detections
        boxes = results.boxes
        num_detections = len(boxes)
        
        # 5. Draw sleek bounding boxes and labels
        for box in boxes:
            # Get box coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class index and confidence score
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            label_text = f"{model.names[cls_idx]} {conf:.2f}"
            
            # Select a modern cyan color (BGR format: Blue, Green, Red)
            color = (255, 191, 0) # Cyan/Blue
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw a sleek label banner
            # Get text size for the background banner
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 0.5 is for font size relative to default of the opencv, 1 is thickness of letters
            
            # Draw semi-transparent background for label text
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1) # -1 is to fill the rectangle completely
            
            # Draw white text over the background banner
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) # the last one smooths the edges of the letters, the color is in BGR format and it gives full white on these values

        # 6. Calculate processing speed (FPS)
        end_time = time.time()
        latency = (end_time - start_time) * 1000 # in milliseconds
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0.0 # frames per second that the computer can process

        # 7. Render on-screen HUD (Heads-Up Display) overlay for a premium look
        # Draw HUD semi-transparent background banner
        cv2.rectangle(frame, (10, 10), (380, 110), (0, 0, 0), -1)
        # Draw a thin border around HUD
        cv2.rectangle(frame, (10, 10), (380, 110), (255, 191, 0), 1)

        # Write text metrics on HUD
        cv2.putText(frame, "AIDL_03 OBJECT RECOGNITION PLATFORM", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA) # 20, 30 is the default starting position of the text
        
        cv2.putText(frame, f"System Status: ACTIVE", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Speed: {fps:.1f} FPS (Latency: {latency:.1f} ms)", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Detections: {num_detections} items in view", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display keyboard tips at the bottom of the screen
        cv2.putText(frame, "Controls: Press [S] to Save Screenshot | Press [Q] to Quit", (20, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("AIDL_03 Object Recognition (YOLOv8)", frame)
        
        # 8. Keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to quit
        if key == ord('q'):
            print("Closing application...")
            break
            
        # Press 's' to save screenshots
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_filename = f"{screenshot_dir}/figure_{screenshot_counter}_raw_{timestamp}.jpg"
            annotated_filename = f"{screenshot_dir}/figure_{screenshot_counter}_detected_{timestamp}.jpg"
            
            # Save raw and annotated frames
            cv2.imwrite(raw_filename, raw_frame)
            cv2.imwrite(annotated_filename, frame)
            
            print(f"Captured Figure {screenshot_counter}:")
            print(f"  - Raw saved to: {raw_filename}")
            print(f"  - Annotated saved to: {annotated_filename}")
            
            # Show a temporary visual flash on screen that screenshot was captured
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]), (255, 255, 255), -1) # [1] is the width of the image, [0] the height
            cv2.imshow("AIDL_03 Object Recognition (YOLOv8)", flash)
            cv2.waitKey(100) # flash for 100ms
            
            screenshot_counter += 1

    # Cleanup hardware and windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
