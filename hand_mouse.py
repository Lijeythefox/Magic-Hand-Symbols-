import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

screen_w, screen_h = pyautogui.size()

smooth_x, smooth_y = screen_w / 2, screen_h / 2
sensitivity = 0.25
sens_min, sens_max = 0.05, 0.6
margin = 0.12

pinch_prev = False
control_active = True
show_preview = True
debug_mode = True
menu_open = False
dragging_slider = False
click_pulses = []

WINDOW_NAME = 'Hand Mouse'
MENU_W, MENU_H = 340, 220


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_range(value, in_min, in_max, out_min, out_max):
    value = max(min(value, in_max), in_min)
    return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


def menu_rects(w, h):
    mx = (w - MENU_W) // 2
    my = (h - MENU_H) // 2
    resume_rect = (mx + 30, my + 55, mx + MENU_W - 30, my + 95)
    slider_track = (mx + 30, my + 135, mx + MENU_W - 30, my + 145)
    checkbox_rect = (mx + 30, my + 175, mx + 50, my + 195)
    return mx, my, resume_rect, slider_track, checkbox_rect


def point_in(rect, x, y):
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def draw_skeleton_plain(frame, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (210, 210, 210), 1, cv2.LINE_AA)
    tip_idx = {4, 8, 12, 16, 20}
    for i, p in enumerate(pts):
        r = 4 if i in tip_idx else 3
        cv2.circle(frame, p, r, (230, 230, 230), 1, cv2.LINE_AA)


def mouse_callback(event, x, y, flags, param):
    global menu_open, control_active, sensitivity, debug_mode, dragging_slider
    if not menu_open:
        return
    frame_w, frame_h = param
    mx, my, resume_rect, slider_track, checkbox_rect = menu_rects(frame_w, frame_h)

    if event == cv2.EVENT_LBUTTONDOWN:
        if point_in(resume_rect, x, y):
            menu_open = False
            control_active = True
        elif point_in((slider_track[0] - 8, slider_track[1] - 8, slider_track[2] + 8, slider_track[3] + 8), x, y):
            dragging_slider = True
            t = map_range(x, slider_track[0], slider_track[2], 0, 1)
            sensitivity = sens_min + t * (sens_max - sens_min)
        elif point_in(checkbox_rect, x, y):
            debug_mode = not debug_mode
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_slider:
            t = map_range(x, slider_track[0], slider_track[2], 0, 1)
            sensitivity = sens_min + t * (sens_max - sens_min)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_slider = False


def draw_menu(frame):
    h, w = frame.shape[:2]
    mx, my, resume_rect, slider_track, checkbox_rect = menu_rects(w, h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.rectangle(frame, (mx, my), (mx + MENU_W, my + MENU_H), (30, 30, 30), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (mx, my), (mx + MENU_W, my + MENU_H), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "PAUSED", (mx + 30, my + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    rx0, ry0, rx1, ry1 = resume_rect
    cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 150, 90), -1, cv2.LINE_AA)
    cv2.putText(frame, "Resume", (rx0 + 90, ry0 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Sensitivity", (mx + 30, my + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    sx0, sy0, sx1, sy1 = slider_track
    cv2.line(frame, (sx0, (sy0 + sy1) // 2), (sx1, (sy0 + sy1) // 2), (120, 120, 120), 3, cv2.LINE_AA)
    t = map_range(sensitivity, sens_min, sens_max, 0, 1)
    handle_x = int(sx0 + t * (sx1 - sx0))
    cv2.circle(frame, (handle_x, (sy0 + sy1) // 2), 8, (0, 180, 255), -1, cv2.LINE_AA)

    cx0, cy0, cx1, cy1 = checkbox_rect
    cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), (220, 220, 220), 1, cv2.LINE_AA)
    if debug_mode:
        cv2.line(frame, (cx0 + 3, (cy0 + cy1) // 2), (cx0 + 8, cy1 - 3), (0, 220, 120), 2, cv2.LINE_AA)
        cv2.line(frame, (cx0 + 8, cy1 - 3), (cx1 - 3, cy0 + 3), (0, 220, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, "Debug mode", (cx1 + 12, cy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, "Tab to close", (mx + 30, my + MENU_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback, (640, 480))

prev_time = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    now = time.time()
    dt = now - prev_time
    prev_time = now

    status = "no hand"

    if control_active:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            wrist = pts[0]
            palm = pts[9]

            if debug_mode and show_preview:
                draw_skeleton_plain(frame, pts)

            tips = [8, 12, 16, 20]
            mcps = [5, 9, 13, 17]
            extended = sum(1 for t, m in zip(tips, mcps) if dist(pts[t], wrist) > dist(pts[m], wrist) * 1.2)
            open_palm = extended >= 3

            scale = dist(wrist, pts[9]) + 1e-6
            pinch_now = dist(pts[4], pts[8]) < scale * 0.5

            if open_palm or pinch_now:
                nx = map_range(palm[0], w * margin, w * (1 - margin), 0, screen_w)
                ny = map_range(palm[1], h * margin, h * (1 - margin), 0, screen_h)
                smooth_x += (nx - smooth_x) * sensitivity
                smooth_y += (ny - smooth_y) * sensitivity
                pyautogui.moveTo(smooth_x, smooth_y)

                color = (0, 140, 255) if pinch_now else (0, 220, 120)
                cv2.circle(frame, palm, 10, color, -1, cv2.LINE_AA)
                status = "click" if pinch_now else "move"
            else:
                status = "hand idle"

            if pinch_now and not pinch_prev:
                pyautogui.click()
                click_pulses.append({'pos': palm, 'age': 0.0})
            pinch_prev = pinch_now
        else:
            pinch_prev = False
    else:
        pinch_prev = False
        status = "paused"

    alive_pulses = []
    for p in click_pulses:
        p['age'] += dt
        if p['age'] < 0.4:
            radius = int(p['age'] * 160)
            cv2.circle(frame, p['pos'], radius, (0, 140, 255), 2, cv2.LINE_AA)
            alive_pulses.append(p)
    click_pulses = alive_pulses

    if debug_mode and show_preview:
        cv2.putText(frame, status, (14, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if menu_open:
        draw_menu(frame)
        cv2.imshow(WINDOW_NAME, frame)
    elif show_preview:
        cv2.imshow(WINDOW_NAME, frame)
    else:
        blank = np.zeros_like(frame)
        cv2.putText(blank, "Preview hidden - h to show, Tab for menu", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, blank)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('h'):
        show_preview = not show_preview
    if key == 9:
        menu_open = not menu_open
        if menu_open:
            control_active = False
            show_preview = True
        else:
            control_active = True

cap.release()
cv2.destroyAllWindows()
