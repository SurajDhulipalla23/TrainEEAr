import cv2
import numpy as np

def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurement variables
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

        # if tip:
        #     # Update the Kalman filter with the measurement
        #     corrected = kalman.correct(np.array(tip, np.float32).reshape(2, 1))
        #     cv2.circle(frame, (corrected[0], corrected[1]), 5, (0, 255, 0), -1)
        #     cv2.line(frame, (int(corrected[0] - 30), int(corrected[1])), (int(corrected[0] + 30), int(corrected[1])), (0, 0, 255), 2)
        
        if tip:
            # Update the Kalman filter with the measurement
            corrected = kalman.correct(np.array(tip, np.float32).reshape(2, 1))
    
            corrected_x = int(corrected[0,0])
            corrected_y = int(corrected[1,0])
    
            cv2.circle(frame, (corrected_x, corrected_y), 5, (0, 255, 0), -1)
            cv2.line(frame, (corrected_x - 30, corrected_y), (corrected_x + 30, corrected_y), (0, 0, 255), 2)


        cv2.imshow('Pencil Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
