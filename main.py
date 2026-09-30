import threading
from posture_detection import run_posture_detection
from app import run_flask

# Thread for detection
t1 = threading.Thread(target=run_posture_detection)

# Thread for dashboard
t2 = threading.Thread(target=run_flask)


t1.start()
t2.start()


t1.join()
t2.join()