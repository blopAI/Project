import cv2
import numpy as np

def apply_low_pass_filter(frame, cutoff_frequency):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_gray = frame_gray.astype(np.float32)  
    frame_freq = cv2.dft(frame_gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    rows, cols = frame_freq.shape[:2]
    mask = np.zeros((rows, cols, 2), np.float32)  
    cutoff_row = int(rows * cutoff_frequency)
    cutoff_col = int(cols * cutoff_frequency)
    mask[:cutoff_row, :cutoff_col] = 1
    
    frame_freq_filtered = frame_freq * mask
    
    frame_filtered = cv2.idft(frame_freq_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    return frame_filtered.astype(np.uint8)

def add_noise_to_frame(frame, noise_level):
    noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
    noisy_frame = frame + noise
    return noisy_frame


video_path = 'downscaledRotated.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Failed to open the video at '{video_path}'.")

cutoff_frequency = 0.05

while cap.isOpened():
    ret, frame = cap.read()
    

    if ret:
        # Apply the low-pass filter to the frame
        filtered_frame = apply_low_pass_filter(frame, cutoff_frequency)

        # Display the original and filtered frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Filtered Frame', filtered_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
