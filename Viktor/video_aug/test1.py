import numpy as np
import cv2


# Open the video file
cap = cv2.VideoCapture('downscaledRotated.mov')
import numpy as np

def add_noise_to_frame(frame, noise_level):
    noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
    noisy_frame = frame + noise
    return noisy_frame


while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Apply modifications to the frame as per the options
        # For example, adding noise to the recording
        noisy_frame = add_noise_to_frame(frame, 45)

        # Send the modified frame to the measurement forwarding system
        # ...

        # Display the modified frame
        cv2.imshow('Frame', frame)
        cv2.imshow('Modified Frame', noisy_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
