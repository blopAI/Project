import cv2

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
 
video_path = 'downscaledRotated.mov'
output_path = 'resized_video.mp4'
width = 640
height = 480

resize_video(video_path, output_path, width, height)
