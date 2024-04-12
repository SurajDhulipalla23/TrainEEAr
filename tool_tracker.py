import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# plt.imshow(frame)
# plt.show()
# cap.release

# while cap.isOpened():
#     ret, frame = cap.read()
#     cv2.imshow('WebCam', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release
# cv2.destroyWindow()


def find_pencil_tip(contours):
    if not contours:
        return None
    
    # Assuming the largest contour is the pencil, which may be true if the pencil is the most prominent object
    c = max(contours, key=cv2.contourArea)
    # Get the topmost point of the contour; this assumes the pencil is oriented top-down in the image.
    # Change the index if the orientation is different.
    return tuple(c[c[:, :, 1].argmin()][0])

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours 
        #Make this step better by instead of automatically assuming the largest contour is the pencil, 
        #Use additional criteria. For example, you could consider the aspect ratio or the shape of the contour.
        #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color range for black (assuming it's dark gray to black for a broader range)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # Define color range for golden (you may need to adjust this based on actual color)
        lower_golden = np.array([20, 100, 100])
        upper_golden = np.array([35, 255, 255])
        mask_golden = cv2.inRange(hsv, lower_golden, upper_golden)

        # Combine the two masks
        mask = cv2.bitwise_or(mask_black, mask_golden)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find pencil tip
        tip = find_pencil_tip(contours)

        if tip:
            cv2.circle(frame, tip, 5, (0, 255, 0), -1)  # draw a green circle at the pencil tip
            cv2.line(frame, (tip[0] - 30, tip[1]), (tip[0] + 30, tip[1]), (0, 0, 255), 2)  # draw a red tangent line

        cv2.imshow('Pencil Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
