import os
import random
import shutil

def random_select_files(source_folder, destination_folder, num_files=150):
    # ตรวจสอบว่าโฟลเดอร์ปลายทางมีหรือไม่ ถ้าไม่มีให้สร้าง
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # ค้นหาไฟล์ทั้งหมดในโฟลเดอร์ต้นทางที่เป็นไฟล์รูปภาพ
    all_files = [f for f in os.listdir(source_folder) 
                 if os.path.isfile(os.path.join(source_folder, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]
    
    # เลือกไฟล์แบบสุ่มจำนวนที่ต้องการ (ถ้าไฟล์มีไม่ถึง 150 จะเลือกทั้งหมด)
    selected_files = random.sample(all_files, min(len(all_files), num_files))
    
    # ย้ายหรือคัดลอกไฟล์ไปยังโฟลเดอร์ปลายทาง
    for file in selected_files:
        shutil.copy2(os.path.join(source_folder, file), os.path.join(destination_folder, file))
        print(f"คัดลอกไฟล์: {file}")

# ตัวอย่างการใช้งาน
source_folder = "./Frames/Unknow"       # เปลี่ยนเป็น path โฟลเดอร์ต้นทาง
destination_folder = "./Test/Unknow/"    # เปลี่ยนเป็น path โฟลเดอร์ปลายทาง
num_files = 150  # จำนวนไฟล์ที่ต้องการ

random_select_files(source_folder, destination_folder, num_files)
