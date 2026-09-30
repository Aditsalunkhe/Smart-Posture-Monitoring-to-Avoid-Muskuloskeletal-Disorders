def run_posture_detection():

    import cv2
    import mediapipe as mp
    import numpy as np
    import csv
    import time
    from datetime import datetime
    from plyer import notification

    # ---------------- INIT ---------------- #
    last_log_time = 0
    LOG_INTERVAL = 10

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # CSV Setup
    csv_file = open('posture_data.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Time", "Neck Angle", "Back Angle", "Posture"])

    # Calibration variables
    calibrated = False
    calibration_frames = 0
    neck_calib = []
    back_calib = []

    CALIBRATION_TIME = 100  # ~5 sec

    # Threshold tolerance
    NECK_TOL = 12
    BACK_TOL = 12

    # Alert system
    bad_posture_start = None
    ALERT_TIME = 5  # seconds

    # ---------------- FUNCTIONS ---------------- #

    def calculate_angle(a, b, c):

        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = (
            np.arctan2(c[1] - b[1], c[0] - b[0])
            - np.arctan2(a[1] - b[1], a[0] - b[0])
        )

        angle = np.abs(radians * 180.0 / np.pi)

        return 360 - angle if angle > 180 else angle

    def classify_posture(neck, back, ref_neck, ref_back):

        neck_diff = neck - ref_neck
        back_diff = back - ref_back

        if abs(neck_diff) < NECK_TOL and abs(back_diff) < BACK_TOL:
            return "GOOD", (0, 255, 0)

        elif neck_diff > NECK_TOL and abs(back_diff) < BACK_TOL:
            return "FORWARD_HEAD", (0, 165, 255)

        elif back_diff > BACK_TOL and abs(neck_diff) < NECK_TOL:
            return "SLOUCHED", (255, 0, 0)

        else:
            return "SLOUCHED+FORWARD_HEAD", (0, 0, 255)

    def send_notification():

        notification.notify(
            title="Posture Alert ⚠",
            message="Fix your posture!",
            timeout=3
        )

    # ---------------- CAMERA ---------------- #

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # Process image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:

                landmarks = results.pose_landmarks.landmark

                # Keypoints
                left_shoulder = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ].y
                ]

                right_shoulder = [
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    ].y
                ]

                nose = [
                    landmarks[
                        mp_pose.PoseLandmark.NOSE.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.NOSE.value
                    ].y
                ]

                left_hip = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_HIP.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_HIP.value
                    ].y
                ]

                left_ear = [
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_EAR.value
                    ].x,
                    landmarks[
                        mp_pose.PoseLandmark.LEFT_EAR.value
                    ].y
                ]

                # Angles
                neck_angle = calculate_angle(
                    left_shoulder,
                    nose,
                    right_shoulder
                )

                back_angle = calculate_angle(
                    left_hip,
                    left_shoulder,
                    left_ear
                )

                # ---------------- CALIBRATION ---------------- #

                if not calibrated:

                    neck_calib.append(neck_angle)
                    back_calib.append(back_angle)

                    calibration_frames += 1

                    cv2.putText(
                        image,
                        "CALIBRATING... SIT STRAIGHT",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    if calibration_frames >= CALIBRATION_TIME:

                        ref_neck = np.mean(neck_calib)
                        ref_back = np.mean(back_calib)

                        calibrated = True

                        print("Calibration Done")
                        print("Ref Neck:", ref_neck)
                        print("Ref Back:", ref_back)

                    continue

                # ---------------- CLASSIFICATION ---------------- #

                posture, color = classify_posture(
                    neck_angle,
                    back_angle,
                    ref_neck,
                    ref_back
                )

                # ---------------- ALERT SYSTEM ---------------- #

                if posture != "GOOD":

                    if bad_posture_start is None:
                        bad_posture_start = time.time()

                    elif time.time() - bad_posture_start > 5:

                        send_notification()

                        bad_posture_start = None  # prevent spam

                else:
                    bad_posture_start = None

                # ---------------- DISPLAY ---------------- #

                cv2.putText(
                    image,
                    f'Neck: {int(neck_angle)}',
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f'Back: {int(back_angle)}',
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f'Status: {posture}',
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # ---------------- CSV LOGGING ---------------- #

                timestamp = datetime.now().strftime("%H:%M:%S")

                csv_writer.writerow([
                    timestamp,
                    round(neck_angle, 2),
                    round(back_angle, 2),
                    posture
                ])

            except:
                pass

            # Draw landmarks

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Show frame

            cv2.imshow(
                'Posture Monitor (Advanced)',
                image
            )

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Cleanup

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()