import cv2
import math as m
import mediapipe as mp
import streamlit as st
import sqlite3
import pandas as pd
import time
import pygame
import datetime
import matplotlib.pyplot as plt  # Importing Matplotlib for custom plotting

# 初始化 Mediapipe 姿勢模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_tracking_confidence=0.5)

# 設定 Streamlit 頁面標題
st.title("姿勢偵測與分析系統")
FRAME_WINDOW = st.image([])

# 初始化 SQLite 資料庫
def init_db():
    conn = sqlite3.connect("posture.db")
    cursor = conn.cursor()

    # 創建新的資料表，torso_inclination欄位為REAL（小數）
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS posture_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cva_angle REAL,
            torso_inclination REAL,
            posture_status TEXT,
            alert_issued_shifted INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

# 儲存姿勢數據到資料庫
def log_posture(cva_angle, torso_inclination, posture_status, alert_issued_shifted):
    conn = sqlite3.connect("posture.db")
    cursor = conn.cursor()
    cursor.execute(''' 
        INSERT INTO posture_data (timestamp, cva_angle, torso_inclination, posture_status, alert_issued_shifted)
        VALUES (datetime('now', 'localtime'), ?, ?, ?, ?)
    ''', (cva_angle, torso_inclination, posture_status, alert_issued_shifted))
    conn.commit()
    conn.close()

# 計算CVA角度
def calculate_cva_angle(ear_x, ear_y, shoulder_x, shoulder_y):
    try:
        delta_x = shoulder_x - ear_x
        delta_y = ear_y - shoulder_y
        angle = m.degrees(m.atan2(delta_y, delta_x))
        return abs(angle)
    except ZeroDivisionError:
        return 90  # 預設安全值

# 計算兩點形成的夾角，並返回小數
def findAngle(x1, y1, x2, y2):
    try:
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * abs(y1)))
        return theta * 180 / m.pi  # 返回小數
    except (ValueError, ZeroDivisionError):
        return 90  # 預設安全值

# 初始化資料庫
init_db()

# 初始化音效相關設定
warning_sound = "Announcement_sound_effect.mp3"
pygame.mixer.init()
is_playing_warning_sound = False

# 播放警告音效
def play_warning_sound():
    global is_playing_warning_sound
    if not is_playing_warning_sound:
        pygame.mixer.music.load(warning_sound)
        pygame.mixer.music.play(loops=-1)
        is_playing_warning_sound = True

# 停止警告音效
def stop_warning_sound():
    global is_playing_warning_sound
    if is_playing_warning_sound:
        pygame.mixer.music.stop()
        is_playing_warning_sound = False

# Main logic
detect_posture = st.sidebar.button("偵測姿勢")
cap = None

if detect_posture:
    if cap and cap.isOpened():
        cap.release()
    FRAME_WINDOW.empty()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    bad_posture_start_time = None
    posture_alert_time = 5  # Alert time for bad posture
    showed_warning = False
    warning_message = None
    previous_posture_status = "Good"  # Initialize as Good
    last_log_time = None  # Initialize for limiting log frequency

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Cannot capture video frame")
            break

        h, w = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            # Get coordinates of ear and shoulder
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Draw the detected points
            cv2.circle(frame, (l_ear_x, l_ear_y), 5, (0, 255, 0), -1)  # Green circle for ear
            cv2.circle(frame, (l_shldr_x, l_shldr_y), 5, (255, 0, 0), -1)  # Blue circle for shoulder
            cv2.circle(frame, (l_hip_x, l_hip_y), 7, (127, 255, 0), -1)
            cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (127, 255, 0), 2)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (127, 255, 0), 2)

            # Calculate CVA angle
            cva_angle = calculate_cva_angle(l_ear_x, l_ear_y, l_shldr_x, l_shldr_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Posture detection logic
            if cva_angle >= 50 and torso_inclination < 15:
                posture_status = "Good"
                alert_issued_shifted = 0  # Good posture
                cv2.putText(frame, "Good Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)
                stop_warning_sound()
                bad_posture_start_time = None
                showed_warning = False
                if warning_message:
                    warning_message.empty()
                    warning_message = None
            else:
                posture_status = "Bad"
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                if time.time() - bad_posture_start_time >= posture_alert_time and not showed_warning:
                    if warning_message is None:
                        warning_message = st.warning("你已經保持不良姿勢超過5秒鐘了！")
                    play_warning_sound()
                    showed_warning = True
                    alert_issued_shifted = 1  # Set alert issued when warning is shown
                else:
                    alert_issued_shifted = 0  # No alert issued yet

                cv2.putText(frame, "Bad Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)

            # 限制每秒鐘只記錄一次姿勢狀態
            current_time = time.time()
            if last_log_time is None or current_time - last_log_time >= 1:  # 每秒鐘記錄一次
                log_posture(cva_angle, torso_inclination, posture_status, alert_issued_shifted)
                last_log_time = current_time

            cv2.putText(frame, f"CVA: {cva_angle:.1f} deg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)
            cv2.putText(frame, f"Torso: {torso_inclination:.1f} deg", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)

        FRAME_WINDOW.image(frame)

    if cap and cap.isOpened():
        cap.release()

# Streamlit button to view historical data
st.sidebar.subheader("查看歷史數據")

# 讓使用者選擇開始和結束日期
start_date = st.sidebar.date_input("開始日期", min_value=datetime.date(2020, 1, 1), max_value=datetime.date.today(), value=datetime.date.today())
end_date = st.sidebar.date_input("結束日期", min_value=start_date, max_value=datetime.date.today(), value=datetime.date.today())

# 顯示方式選項
display_mode = st.sidebar.radio("顯示方式", ["長條圖", "數據表"])

# 讀取資料庫數據
conn = sqlite3.connect("posture.db")
query = "SELECT * FROM posture_data"
data = pd.read_sql_query(query, conn)
conn.close()

# 時間戳處理與篩選
data["timestamp"] = pd.to_datetime(data["timestamp"])

# 根據使用者選擇的日期範圍過濾數據
start_time = datetime.datetime.combine(start_date, datetime.datetime.min.time())
end_time = datetime.datetime.combine(end_date, datetime.datetime.max.time())

filtered_data = data[(data["timestamp"] >= start_time) & (data["timestamp"] <= end_time)]

# 篩選出不良姿勢的數據
bad_posture_data = filtered_data[filtered_data["posture_status"] == "Bad"]

# 提取小時（將時間戳格式化為 YYYY-MM-DD HH）
bad_posture_data["hour"] = bad_posture_data["timestamp"].dt.strftime('%Y-%m-%d %H')

# 計算每小時內 "Bad Posture" 的次數
bad_posture_count = bad_posture_data.groupby('hour').size()

# 根據顯示方式顯示數據
if display_mode == "長條圖":
    # Create a Matplotlib figure for the bar chart
    fig, ax = plt.subplots()

    # Plot the bar chart
    ax.bar(bad_posture_count.index, bad_posture_count.values, color='b')

    # Set x-axis and y-axis labels
    ax.set_xlabel("Date and Hour")  # X-axis label
    ax.set_ylabel("Bad Posture Count")  # Y-axis label

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Display the chart in Streamlit
    st.pyplot(fig)

elif display_mode == "數據表":
    # 顯示詳細數據表
    st.write(filtered_data[["timestamp", "cva_angle", "torso_inclination", "posture_status"]])
