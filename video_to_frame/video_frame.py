import cv2

def extract_frames(input_video, output_folder, start_time, end_time, frame_interval=1):
    print('extracting frames from ' + input_video + ' starting ' + str(start_time) + 's')
    # Open the video file
    video_capture = cv2.VideoCapture(input_video)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frame numbers based on time
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)
    print('start_frame is ' + str(start_frame) + ' & end_frame is ' + str(end_frame))

    # Set the video capture object to the start frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create the output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read and save frames
    frame_count = start_frame
    while frame_count <= end_frame:
        ret, frame = video_capture.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(str((end_frame - start_frame) / frame_interval) + ' frames extracted to ' + output_folder)

if __name__ == "__main__":
    # Example usage
    input_video = "ue/20231212(1)/20231212/GX010353.MP4"
    output_folder = "frame/GX010353"
    start_time = 10.  # Start time in seconds
    end_time = 11.5    # End time in seconds
    frame_interval = 5  # Save every 10th frame

    extract_frames(input_video, output_folder, start_time, end_time, frame_interval)
