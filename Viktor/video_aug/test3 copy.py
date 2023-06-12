import cv2
import numpy as np
import numpy as np
import cv2
import threading
import redis
import signal
import time
import kafka



def conv_2d(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    H, W, _ = slika.shape
    N, M = jedro.shape

    pad_H = N // 2
    pad_W = M // 2
    padding = ((pad_H, pad_H), (pad_W, pad_W), (0, 0))
    slika_padded = np.pad(slika, padding, mode='constant')

    izhod = np.zeros(slika.shape, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            subarray = slika_padded[i:i+N, j:j+M]
            elementwise_product = subarray * jedro
            output = np.sum(elementwise_product)
            izhod[i, j] = output

    return izhod

def RGB_glajenje(slika: np.ndarray, faktor: float) -> np.ndarray:
    jedro = np.ones((3, 3), dtype=np.float32) / 9
    glajena_slika = conv_2d(slika.astype(np.float32), jedro)
    faktor_s = faktor / 9
    glajena_slika = slika.astype(np.float32) * (1 - faktor_s) + glajena_slika * faktor_s
    glajena_slika = np.clip(glajena_slika, 0, 255).astype(np.uint8)
    return glajena_slika

def grey_scale_sharp(slika: np.ndarray, faktor_glajenja: float, faktor_ostrenja: float) -> np.ndarray:
    jedro_ostrenje = np.array([[-1, -1, -1], [-1, 9 + faktor_ostrenja, -1], [-1, -1, -1]], dtype=np.float32)
    glajena_slika = RGB_glajenje(slika, faktor_glajenja)
    ostrena_slika = conv_2d(glajena_slika.astype(np.float32), jedro_ostrenje)
    ostrena_slika = np.clip(ostrena_slika, 0, 255).astype(np.uint8)
    return ostrena_slika

def apply_low_pass_filter(frame, cutoff_frequency):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_gray = frame 
    frame_gray = frame_gray.astype(np.float32)  
    frame_freq = cv2.dft(frame_gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    #frame_freq = cv2.dft(frame_gray)
    
    rows, cols = frame_freq.shape[:2]
    mask = np.zeros((rows, cols, 2), np.float32)  
    cutoff_row = int(rows * cutoff_frequency)
    cutoff_col = int(cols * cutoff_frequency)
    mask[:cutoff_row, :cutoff_col] = 1
    
    frame_freq_filtered = frame_freq * mask
    
    frame_filtered = cv2.idft(frame_freq_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    return frame_filtered.astype(np.uint8)

def apply_low_pass_filter2(frame, cutoff_frequency, sampling_frequency):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_gray = frame_gray.astype(np.float32)  
    frame_freq = cv2.dft(frame_gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    rows, cols = frame_freq.shape[:2]
    mask = np.zeros((rows, cols, 2), np.float32)  
    cutoff_row = int(rows * cutoff_frequency)
    cutoff_col = int(cols * cutoff_frequency)
    mask[:cutoff_row, :cutoff_col] = 1
    
    mask = cv2.resize(mask, (int(cols * sampling_frequency), int(rows * sampling_frequency)))
    
    frame_freq_filtered = frame_freq * mask
    
    frame_filtered = cv2.idft(frame_freq_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    return frame_filtered.astype(np.uint8)


def add_noise_to_frame(frame, noise_level):
    noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
    noisy_frame = frame + noise
    return noisy_frame

def add_noise(image, strength):
    row, col, ch = image.shape
    mean = 0
    var = strength * 255
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def resize_video(video_path, output_path, width, height):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open the video at '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Resize the frame
            resized_frame = cv2.resize(frame, (width, height))

            # Write the resized frame to the output video
            output_video.write(resized_frame)

            # Display the resized frame (optional)
            cv2.imshow('Resized Frame', resized_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture and writer, and close the windows
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def change_sampling_rate(input_video_path, new_sampling_rate):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate the frame interval for the new sampling rate
    new_frame_interval = round(fps / new_sampling_rate)

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_counter % new_frame_interval == 0:
            # Display the frame with the new sampling rate
            cv2.imshow('Processed Frame', frame)

        frame_counter += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_path = 'testingv.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Failed to open the video at '{video_path}'.")

effect_choice = int(input("Choose an effect:\n1. Add Noise\n2. Add Low Pass Filter\n3. Greyscale Sharpening(very slow)\n4. Resize Video\n5. Test filter, vzorcevalna\n6. Noise - version2\n7. Sampling rate\n"))

if effect_choice == 1:
    noise_level = float(input("Enter noise level (0-100): "))
elif effect_choice == 2:
    cutoff_frequency = float(input("Enter cutoff frequency (0-1): "))
elif effect_choice == 3:
    val1 = float(input("Enter (1)"))
    val2 = float(input("Enter loss(~1)"))
elif effect_choice == 4:
    output_path = "test.mp4"
    WIDTH = float(input("Enter your new WIDTH: "))
    HEIGHT = float(input("Enter your new HEIGHT: "))
elif effect_choice == 5:
    cutoff_frequency = float(input("Enter cutoff frequency (0-1): "))
    sampling_frequency = float(input("Test: "))
elif effect_choice == 6:
    strengthv = float(input("Enter noise level (0.1-1.0): "))
elif effect_choice == 7:
    sampler = float(input("Enter sample rate: "))

else:
    raise ValueError("Invalid effect choice.")

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        if effect_choice == 1:
            processed_frame = add_noise_to_frame(frame, noise_level)
        elif effect_choice == 2:
            processed_frame = apply_low_pass_filter(frame, cutoff_frequency)
        elif effect_choice == 3:
            processed_frame = grey_scale_sharp(frame, val1, val2)
        elif effect_choice == 4:
            processed_frame = resize_video(frame, output_path, WIDTH, HEIGHT)
        elif effect_choice == 5:
            processed_frame = apply_low_pass_filter2(frame, cutoff_frequency, sampling_frequency)
        elif effect_choice == 6:
            processed_frame = add_noise(frame, strength=strengthv)
        elif effect_choice == 7:
            change_sampling_rate(video_path, new_sampling_rate=sampler)
            break
        else:
            raise ValueError("Invalid effect choice.")
        
        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()