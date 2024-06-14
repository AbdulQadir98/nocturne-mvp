import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='live')
    parser.add_argument(
        "--webcam-resolution",
        default=[1720,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def enhance(frame):
    equalized_channels = [cv2.equalizeHist(channel) for channel in cv2.split(frame)]
    equalized_image = cv2.merge(equalized_channels)
    # Experiment with different values
    gamma = 12  
    corrected_image = np.power(equalized_image / 255.0, gamma)
    img = np.uint8(corrected_image * 255)
    return img
    
def downscale_frame(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    downscaled_frame = cv2.cvtColor(grey_frame, cv2.COLOR_GRAY2RGB)
    return downscaled_frame

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO('models/best.pt')

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=2,
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, classes=0)[0]
        detections = sv.Detections.from_ultralytics(result)

        # enhanced_frame = enhance(frame)

        frame = box_annotator.annotate(frame, detections)

        cv2.imshow('yolov8', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
