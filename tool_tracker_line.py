import cv2
import numpy as np
import time

def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2) 
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) 
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-5
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-4
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    kalman.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)
    return kalman

def find_pencil_tip(contours):
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return tuple(c[c[:, :, 1].argmin()][0])

def main():
    cap = cv2.VideoCapture(0)
    kalman = initialize_kalman()
    
    path = []  
    start_time = None  # Store when the pencil was first detected

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tip = find_pencil_tip(contours)

        # Predict the next state
        predicted = kalman.predict()

        if tip:
            if start_time is None:
                start_time = time.time()

            # Update the Kalman filter with the measurement
            corrected = kalman.correct(np.array(tip, np.float32).reshape(2, 1))
            
            corrected_x = int(corrected[0,0])
            corrected_y = int(corrected[1,0])
            
            if time.time() - start_time > 5:
                path.append((corrected_x, corrected_y))
                cv2.circle(frame, (corrected_x, corrected_y), 5, (0, 255, 0), -1)
                cv2.line(frame, (corrected_x - 30, corrected_y), (corrected_x + 30, corrected_y), (0, 0, 255), 2)

        # Draw the path
        if start_time and time.time() - start_time > 5:
            for i in range(1, len(path)):
                cv2.line(frame, path[i - 1], path[i], (255, 0, 0), 2)  # Blue color for path

        cv2.imshow('Tool Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
