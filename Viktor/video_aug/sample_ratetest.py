import cv2

def change_sampling_rate(input_video_path, output_video_path, new_sampling_rate):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate the frame interval for the new sampling rate
    new_frame_interval = round(fps / new_sampling_rate)

    # Create VideoWriter object to write the modified video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, new_sampling_rate, (int(cap.get(3)), int(cap.get(4))))

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_counter % new_frame_interval == 0:
            # Write the frame to the output video
            out.write(frame)

        frame_counter += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
input_video = 'testingv.mp4'
output_video = 'output_video.mp4'
new_sampling_rate = 5

change_sampling_rate(input_video, output_video, new_sampling_rate)

