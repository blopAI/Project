import cv2
import numpy as np

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

video_path = 'downscaledRotated.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Failed to open the video at '{video_path}'.")

faktor_glajenja = 0.5
faktor_ostrenja = 1.0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Apply the grey scale sharpening effect to the frame
        processed_frame = grey_scale_sharp(frame, faktor_glajenja, faktor_ostrenja)

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
