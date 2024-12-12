import cv2
import os

# เส้นทางไฟล์วิดีโอ
video_path = './vdo/20241211_105632000_iOS.MOV'  # เปลี่ยนเป็นไฟล์วิดีโอของคุณ
output_dir = './save/Oat/'     # โฟลเดอร์สำหรับบันทึกภาพ

# สร้างโฟลเดอร์สำหรับบันทึกภาพ หากยังไม่มี
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# เปิดวิดีโอ
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าวิดีโอเปิดได้หรือไม่
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# อ่านค่า FPS จากวิดีโอ
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS ของวิดีโอ: {fps}")

# กำหนดจำนวนรูปต่อวินาทีที่ต้องการ
desired_frames_per_second = 5
frame_interval = fps // desired_frames_per_second  # คำนวณความถี่ในการบันทึกเฟรม

print(f"บันทึกทุก {frame_interval} เฟรม เพื่อให้ได้ {desired_frames_per_second} รูปต่อวินาที")

frame_count = 0  # ตัวนับจำนวนเฟรม
saved_count = 0  # ตัวนับเฟรมที่บันทึก

# อ่านวิดีโอเฟรมต่อเฟรม
while True:
    ret, frame = cap.read()

    # หยุดเมื่อวิดีโอจบ
    if not ret:
        break

    # บันทึกเฟรมทุก frame_interval
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"บันทึก: {frame_filename}")
        saved_count += 1

    frame_count += 1

# ปิดการอ่านวิดีโอ
cap.release()
print("บันทึกภาพเสร็จสิ้น")
