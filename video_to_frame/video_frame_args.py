import cv2
import sys
import os

#使い方
#python extract_frames.py input_video.mp4 output_frames_folder 10 20 10

def extract_frames(input_video, output_folder, start_time, end_time, frame_interval=1):
    # Open the video file
    video_capture = cv2.VideoCapture(input_video)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frame numbers based on time
    start_frame = int(start_time * fps)
    end_frame = min(int(end_time * fps), total_frames - 1)

    # Set the video capture object to the start frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read and save frames
    frame_count = 0
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

if __name__ == "__main__":
    # Check for correct number of command line arguments
    if len(sys.argv) != 6:
        print("Usage: python extract_frames.py <input_video> <output_folder> <start_time> <end_time> <frame_interval>")
        sys.exit(1)

    # Parse command line arguments
    input_video = sys.argv[1]
    output_folder = sys.argv[2]
    start_time = float(sys.argv[3])
    end_time = float(sys.argv[4])
    frame_interval = int(sys.argv[5])

    # Perform frame extraction
    extract_frames(input_video, output_folder, start_time, end_time, frame_interval)
