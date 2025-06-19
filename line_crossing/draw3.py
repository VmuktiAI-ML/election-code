import cv2
import json

# File to save line coordinates
json_file = "line_coordinates2.json"

# RTSP video stream (Change this to your RTSP source)
#video_source = "rtsp://admin:bhargav%40123456@192.168.5.9/Streaming/Channels/1"
#video_source ="rtmp://ptz.vmukti.com:80/live-record/VSPL-138150-RAOAR"
video_source = "rtmp://kolkataele2025.vmukti.com:80/live-record/VSPL-148331-OJGFJ"
# video_source = "rtsp://admin:@192.168.5.221:554/ch0_0.264"
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Error: Could not open RTSP stream {video_source}")
    exit()

line_points = []
drawing_line = False
current_line_point = None

def mouse_callback(event, x, y, flags, param):
    global line_points, drawing_line, current_line_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_line = True
        current_line_point = (x, y)
        line_points.clear()
        line_points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and drawing_line:
        current_line_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_line = False
        line_points.append((x, y))

        # Save the coordinates to a JSON file
        if len(line_points) == 2:
            with open(json_file, "w") as f:
                json.dump({"line": line_points}, f)
            print(f"Line coordinates saved: {line_points}")

# Make the window resizable but don't resize the frame
cv2.namedWindow("Draw Line", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Draw Line", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from RTSP stream.")
        break

    display_frame = frame.copy()

    if line_points and current_line_point:
        cv2.line(display_frame, line_points[0], current_line_point, (0, 255, 255), 2)

    cv2.imshow("Draw Line", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

