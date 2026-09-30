import cv2
import mediapipe as mp
import numpy as np
import psutil
import time
import csv

# ----------------------------
# Initialize MediaPipe
# ----------------------------
mp_pose = mp.solutions.pose

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

# ----------------------------
# Statistics Variables
# ----------------------------
frame_count = 0
successful_detections = 0

fps_list = []
cpu_list = []
ram_list = []
proc_time_list = []

start_program_time = time.time()

# CSV Log File
csv_file = open("performance_log.csv", mode="w", newline="")
writer = csv.writer(csv_file)

writer.writerow([
    "Frame",
    "FPS",
    "CPU_Usage(%)",
    "RAM_Usage(%)",
    "RAM_MB",
    "Processing_Time(ms)",
    "Pose_Detected"
])

# ----------------------------
# Pose Estimation
# ----------------------------
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

    while cap.isOpened():

        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # RGB conversion
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose Processing
        results = pose.process(image)

        pose_detected = 0

        if results.pose_landmarks:
            successful_detections += 1
            pose_detected = 1

        # -------------------------
        # Performance Measurements
        # -------------------------

        frame_end = time.time()

        processing_time = (frame_end - frame_start) * 1000

        fps = 1 / (frame_end - frame_start)

        cpu_usage = psutil.cpu_percent()

        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_mb = ram.used / (1024 * 1024)

        # Store values
        fps_list.append(fps)
        cpu_list.append(cpu_usage)
        ram_list.append(ram_percent)
        proc_time_list.append(processing_time)

        # Write CSV
        writer.writerow([
            frame_count,
            round(fps, 2),
            cpu_usage,
            ram_percent,
            round(ram_mb, 2),
            round(processing_time, 2),
            pose_detected
        ])

        # -------------------------
        # Display
        # -------------------------

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0), 2)

        cv2.putText(frame, f"CPU: {cpu_usage:.1f}%",
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),2)

        cv2.putText(frame, f"RAM: {ram_percent:.1f}%",
                    (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),2)

        cv2.imshow("Performance Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ----------------------------
# Final Statistics
# ----------------------------

total_runtime = time.time() - start_program_time

avg_fps = np.mean(fps_list)
avg_cpu = np.mean(cpu_list)
avg_ram = np.mean(ram_list)
avg_proc = np.mean(proc_time_list)

detection_accuracy = (
    successful_detections / frame_count
) * 100

print("\n========= PERFORMANCE REPORT =========")

print(f"Total Runtime            : {total_runtime:.2f} sec")
print(f"Total Frames             : {frame_count}")
print(f"Pose Detection Rate      : {detection_accuracy:.2f}%")

print(f"\nAverage FPS              : {avg_fps:.2f}")
print(f"Maximum FPS              : {max(fps_list):.2f}")
print(f"Minimum FPS              : {min(fps_list):.2f}")

print(f"\nAverage CPU Usage        : {avg_cpu:.2f}%")
print(f"Maximum CPU Usage        : {max(cpu_list):.2f}%")

print(f"\nAverage RAM Usage        : {avg_ram:.2f}%")
print(f"Maximum RAM Usage        : {max(ram_list):.2f}%")

print(f"\nAverage Processing Time  : {avg_proc:.2f} ms")

print("\nResults saved to:")
print("performance_log.csv")

csv_file.close()
cap.release()
cv2.destroyAllWindows()