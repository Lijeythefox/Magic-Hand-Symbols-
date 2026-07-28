import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

particles = []
web_shots = []
pinch_prev = {}
spiderman_prev = {}
ring_time = 0.0
debug_mode = True
status_message = ""
status_timer = 0.0


def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_scale(pts):
    return dist(pts[0], pts[9]) + 1e-6


def is_open_palm(pts):
    wrist = pts[0]
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    extended = 0
    for t, m in zip(tips, mcps):
        if dist(pts[t], wrist) > dist(pts[m], wrist) * 1.35:
            extended += 1
    return extended >= 3


def is_fist(pts):
    wrist = pts[0]
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    curled = 0
    for t, m in zip(tips, mcps):
        if dist(pts[t], wrist) < dist(pts[m], wrist) * 1.05:
            curled += 1
    return curled >= 3


def is_pinch(pts):
    scale = hand_scale(pts)
    return dist(pts[4], pts[8]) < scale * 0.45


def is_spiderman(pts):
    wrist = pts[0]
    index_ext = dist(pts[8], wrist) > dist(pts[5], wrist) * 1.3
    pinky_ext = dist(pts[20], wrist) > dist(pts[17], wrist) * 1.3
    middle_curl = dist(pts[12], wrist) < dist(pts[9], wrist) * 1.05
    ring_curl = dist(pts[16], wrist) < dist(pts[13], wrist) * 1.05
    return index_ext and pinky_ext and middle_curl and ring_curl


def fist_angle_deg(pts):
    wrist = pts[0]
    mid_mcp = pts[9]
    dx = mid_mcp[0] - wrist[0]
    dy = wrist[1] - mid_mcp[1]
    return math.degrees(math.atan2(dy, dx))


def near_diagonal(angle):
    a = abs(angle) % 180
    targets = [45, 135]
    return any(abs(a - t) < 20 for t in targets)


def spawn_burst(x, y, color=(60, 160, 255), sharp_color=(140, 210, 255)):
    for _ in range(30):
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(3, 9)
        particles.append({
            'x': x, 'y': y,
            'vx': math.cos(angle) * speed,
            'vy': math.sin(angle) * speed,
            'life': 1.0,
            'size': random.uniform(2, 4),
            'color': color,
            'sharp_color': sharp_color
        })


def spawn_web_shot(origin, target, scale):
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    length = math.hypot(dx, dy) + 1e-6
    direction = (dx / length, dy / length)
    max_dist = 420.0
    speed = max_dist / 0.35
    web_shots.append({
        'origin': origin,
        'dir': direction,
        'traveled': 0.0,
        'max_dist': max_dist,
        'speed': speed,
        'state': 'flying',
        'web_age': 0.0,
        'end': None,
        'scale': scale
    })


def update_and_draw_particles(glow, sharp):
    global particles
    alive = []
    for p in particles:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['vx'] *= 0.95
        p['vy'] *= 0.95
        p['life'] -= 0.02
        if p['life'] > 0:
            r = max(int(p['size'] * p['life']) + 1, 1)
            cv2.circle(glow, (int(p['x']), int(p['y'])), r * 3, p['color'], -1, cv2.LINE_AA)
            cv2.circle(sharp, (int(p['x']), int(p['y'])), r, p['sharp_color'], -1, cv2.LINE_AA)
            alive.append(p)
    particles = alive


def draw_web(glow, sharp, cx, cy, radius, fade, spokes=8, rings=4):
    if radius < 2:
        return
    base = int(245 * fade)
    color_glow = (base, base, base)
    color_sharp = (min(int(255 * fade), 255), min(int(255 * fade), 255), min(int(255 * fade), 255))
    spoke_pts = []
    for i in range(spokes):
        a = i * (2 * math.pi / spokes)
        x1 = int(cx + math.cos(a) * radius)
        y1 = int(cy + math.sin(a) * radius)
        spoke_pts.append((x1, y1))
        cv2.line(glow, (cx, cy), (x1, y1), color_glow, 4, cv2.LINE_AA)
        cv2.line(sharp, (cx, cy), (x1, y1), color_sharp, 1, cv2.LINE_AA)
    for r_i in range(1, rings + 1):
        rr = radius * r_i / rings
        ring_pts = []
        for i in range(spokes):
            a = i * (2 * math.pi / spokes)
            ring_pts.append((int(cx + math.cos(a) * rr), int(cy + math.sin(a) * rr)))
        for i in range(spokes):
            p0 = ring_pts[i]
            p1 = ring_pts[(i + 1) % spokes]
            cv2.line(glow, p0, p1, color_glow, 3, cv2.LINE_AA)
            cv2.line(sharp, p0, p1, color_sharp, 1, cv2.LINE_AA)


def update_and_draw_web_shots(glow, sharp, dt):
    global web_shots
    alive = []
    expand_duration = 0.35
    hold = 1.0
    fade_duration = 1.0
    total = expand_duration + hold + fade_duration
    for shot in web_shots:
        if shot['state'] == 'flying':
            shot['traveled'] = min(shot['traveled'] + shot['speed'] * dt, shot['max_dist'])
            cur = (
                int(shot['origin'][0] + shot['dir'][0] * shot['traveled']),
                int(shot['origin'][1] + shot['dir'][1] * shot['traveled'])
            )
            cv2.line(glow, shot['origin'], cur, (245, 245, 245), 5, cv2.LINE_AA)
            cv2.line(sharp, shot['origin'], cur, (255, 255, 255), 2, cv2.LINE_AA)
            if shot['traveled'] >= shot['max_dist']:
                shot['state'] = 'web'
                shot['end'] = cur
            alive.append(shot)
        else:
            shot['web_age'] += dt
            age = shot['web_age']
            max_radius = shot['scale'] * 2.0
            if age < expand_duration:
                radius = max_radius * (age / expand_duration)
                fade = 1.0
            elif age < expand_duration + hold:
                radius = max_radius
                fade = 1.0
            elif age < total:
                radius = max_radius
                fade = 1.0 - ((age - expand_duration - hold) / fade_duration)
            else:
                continue
            draw_web(glow, sharp, shot['end'][0], shot['end'][1], radius, fade)
            alive.append(shot)
    web_shots = alive


def draw_skeleton(glow, sharp, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(glow, pts[a], pts[b], (255, 235, 200), 6, cv2.LINE_AA)
        cv2.line(sharp, pts[a], pts[b], (255, 245, 225), 2, cv2.LINE_AA)
    tip_idx = {4, 8, 12, 16, 20}
    for i, p in enumerate(pts):
        r = 7 if i in tip_idx else 5
        cv2.circle(glow, p, r + 4, (255, 235, 200), -1, cv2.LINE_AA)
        cv2.circle(sharp, p, r, (10, 12, 18), -1, cv2.LINE_AA)
        cv2.circle(sharp, p, r, (255, 250, 240), 2, cv2.LINE_AA)


def draw_ring_marks(glow, sharp, cx, cy, base_r, inner_r, t):
    color_outer = (40, 170, 255)
    color_inner = (255, 190, 90)

    for i in range(36):
        if i % 3 == 0:
            continue
        a0 = t * 0.6 + i * (math.pi * 2 / 36)
        a1 = a0 + (math.pi * 2 / 36) * 0.6
        x0, y0 = int(cx + math.cos(a0) * base_r), int(cy + math.sin(a0) * base_r)
        x1, y1 = int(cx + math.cos(a1) * base_r), int(cy + math.sin(a1) * base_r)
        cv2.line(glow, (x0, y0), (x1, y1), color_outer, 5, cv2.LINE_AA)
        cv2.line(sharp, (x0, y0), (x1, y1), (140, 210, 255), 2, cv2.LINE_AA)

    for i in range(24):
        if i % 2 == 0:
            continue
        a0 = -t * 1.3 + i * (math.pi * 2 / 24)
        a1 = a0 + (math.pi * 2 / 24) * 0.5
        x0, y0 = int(cx + math.cos(a0) * inner_r), int(cy + math.sin(a0) * inner_r)
        x1, y1 = int(cx + math.cos(a1) * inner_r), int(cy + math.sin(a1) * inner_r)
        cv2.line(glow, (x0, y0), (x1, y1), color_inner, 5, cv2.LINE_AA)
        cv2.line(sharp, (x0, y0), (x1, y1), (255, 210, 150), 2, cv2.LINE_AA)

    for i in range(12):
        a = t * 1.8 + i * (math.pi * 2 / 12)
        x0, y0 = int(cx + math.cos(a) * inner_r), int(cy + math.sin(a) * inner_r)
        x1, y1 = int(cx + math.cos(a) * base_r * 0.92), int(cy + math.sin(a) * base_r * 0.92)
        cv2.line(glow, (x0, y0), (x1, y1), color_inner, 3, cv2.LINE_AA)
        cv2.line(sharp, (x0, y0), (x1, y1), (255, 220, 170), 1, cv2.LINE_AA)


def draw_portal(frame, cx, cy, base_r, t):
    h, w = frame.shape[:2]
    r = int(base_r)
    r = min(r, cx, w - cx, cy, h - cy)
    if r < 20:
        return
    x0, y0 = cx - r, cy - r
    size = r * 2
    patch = frame[y0:y0 + size, x0:x0 + size].copy()
    if patch.shape[0] != size or patch.shape[1] != size:
        return

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - r
    dy = yy - r
    dist_from_center = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    swirl = 2.6 * (1 - np.clip(dist_from_center / r, 0, 1)) + t * 0.8
    theta_new = theta + swirl
    map_x = (r + dist_from_center * np.cos(theta_new)).astype(np.float32)
    map_y = (r + dist_from_center * np.sin(theta_new)).astype(np.float32)
    warped = cv2.remap(patch, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    tint = np.zeros_like(warped)
    tint[:] = (40, 90, 255)
    warped = cv2.addWeighted(warped, 0.75, tint, 0.25, 0)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1, cv2.LINE_AA)
    mask3 = cv2.merge([mask, mask, mask])

    roi = frame[y0:y0 + size, x0:x0 + size]
    blended = np.where(mask3 > 0, warped, roi)
    frame[y0:y0 + size, x0:x0 + size] = blended


def draw_mystic_circle(frame, glow, sharp, cx, cy, scale, t):
    base_r = int(scale * 2.6)
    inner_r = int(base_r * 0.7)
    draw_ring_marks(glow, sharp, cx, cy, base_r, inner_r, t)


def draw_claws(glow, sharp, pts, angle_deg):
    angle_rad = math.radians(angle_deg)
    direction = (math.cos(angle_rad), -math.sin(angle_rad))
    scale = hand_scale(pts)
    length = int(scale * 1.9)
    for base_idx in (5, 9, 13):
        bx, by = pts[base_idx]
        ex = int(bx + direction[0] * length)
        ey = int(by + direction[1] * length)
        cv2.line(glow, (bx, by), (ex, ey), (200, 240, 255), 7, cv2.LINE_AA)
        cv2.line(sharp, (bx, by), (ex, ey), (240, 250, 255), 2, cv2.LINE_AA)


def draw_tether(glow, sharp, p1, p2, t):
    steps = 24
    for i in range(steps):
        f0 = i / steps
        f1 = (i + 1) / steps
        wob = math.sin(t * 6 + i) * 4
        x0 = int(p1[0] + (p2[0] - p1[0]) * f0)
        y0 = int(p1[1] + (p2[1] - p1[1]) * f0 + wob)
        x1 = int(p1[0] + (p2[0] - p1[0]) * f1)
        y1 = int(p1[1] + (p2[1] - p1[1]) * f1 + wob)
        cv2.line(glow, (x0, y0), (x1, y1), (255, 210, 120), 6, cv2.LINE_AA)
        cv2.line(sharp, (x0, y0), (x1, y1), (255, 235, 190), 2, cv2.LINE_AA)


prev_time = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]

    now = time.time()
    dt = now - prev_time
    ring_time += dt
    prev_time = now

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    frame = (frame * 0.55).astype(np.uint8)
    glow = np.zeros_like(frame)
    sharp = np.zeros_like(frame)

    active_gestures = []
    hands_data = []

    if results.multi_hand_landmarks:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            pts = [to_px(p, w, h) for p in lm.landmark]

            if debug_mode:
                draw_skeleton(glow, sharp, pts)

            scale = hand_scale(pts)
            palm = pts[9]
            pinch_now = is_pinch(pts)
            pinch_point = ((pts[4][0] + pts[8][0]) // 2, (pts[4][1] + pts[8][1]) // 2)

            if is_open_palm(pts):
                draw_mystic_circle(frame, glow, sharp, palm[0], palm[1], scale, ring_time)
                active_gestures.append('circle')

            if is_fist(pts):
                angle = fist_angle_deg(pts)
                if near_diagonal(angle):
                    draw_claws(glow, sharp, pts, angle)
                    active_gestures.append('claws')

            spiderman_now = is_spiderman(pts)
            if spiderman_now and not spiderman_prev.get(label, False):
                spawn_web_shot(pts[8], pts[8], scale)
                web_shots[-1]['origin'] = pts[0]
                web_shots[-1]['dir'] = (
                    (pts[8][0] - pts[0][0]) / (dist(pts[8], pts[0]) + 1e-6),
                    (pts[8][1] - pts[0][1]) / (dist(pts[8], pts[0]) + 1e-6)
                )
                active_gestures.append('web-shot')
            spiderman_prev[label] = spiderman_now

            if pinch_now and not pinch_prev.get(label, False):
                spawn_burst(pinch_point[0], pinch_point[1])
                active_gestures.append('spark')
            pinch_prev[label] = pinch_now

            hands_data.append({'label': label, 'pinch': pinch_now, 'pinch_point': pinch_point})

    if len(hands_data) == 2 and hands_data[0]['pinch'] and hands_data[1]['pinch']:
        draw_tether(glow, sharp, hands_data[0]['pinch_point'], hands_data[1]['pinch_point'], ring_time)
        active_gestures.append('tether')

    update_and_draw_particles(glow, sharp)
    update_and_draw_web_shots(glow, sharp, dt)

    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=14, sigmaY=14)
    frame = cv2.add(frame, glow)
    frame = cv2.add(frame, sharp)

    if debug_mode:
        label_text = ' + '.join(active_gestures) if active_gestures else 'awaiting gesture'
        cv2.putText(frame, label_text, (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if status_timer > 0:
        status_timer -= dt
        cv2.putText(frame, status_message, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Mystic Hands', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 9:
        debug_mode = not debug_mode
        status_message = "Debug mode activated" if debug_mode else "Debug mode deactivated"
        status_timer = 1.5

cap.release()
cv2.destroyAllWindows()