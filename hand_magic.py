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

lightning_bolts = []
web_shots = []
portals = []
pinch_prev = {}
spiderman_prev = {}
circle_buffer = {}
circle_cooldown = {}
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


def is_draw_pose(pts):
    wrist = pts[0]
    index_ext = dist(pts[8], wrist) > dist(pts[5], wrist) * 1.3
    middle_ext = dist(pts[12], wrist) > dist(pts[9], wrist) * 1.3
    ring_curl = dist(pts[16], wrist) < dist(pts[13], wrist) * 1.05
    pinky_curl = dist(pts[20], wrist) < dist(pts[17], wrist) * 1.05
    return index_ext and middle_ext and ring_curl and pinky_curl


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


def generate_bolt_set(x, y):
    bolts = []
    for _ in range(6):
        angle = random.uniform(0, math.pi * 2)
        length = random.uniform(40, 95)
        segments = []
        cx, cy = x, y
        steps = 5
        for _ in range(steps):
            nx = cx + math.cos(angle) * (length / steps) + random.uniform(-9, 9)
            ny = cy + math.sin(angle) * (length / steps) + random.uniform(-9, 9)
            segments.append(((int(cx), int(cy)), (int(nx), int(ny))))
            cx, cy = nx, ny
        bolts.append(segments)
    return {'bolts': bolts, 'age': 0.0}


def spawn_lightning(x, y):
    lightning_bolts.append(generate_bolt_set(x, y))


def update_and_draw_lightning(glow, sharp, dt):
    global lightning_bolts
    alive = []
    life = 0.22
    for b in lightning_bolts:
        b['age'] += dt
        if b['age'] < life:
            fade = 1 - (b['age'] / life)
            glow_color = (int(255 * fade), int(170 * fade), int(60 * fade))
            sharp_color = (int(255 * fade), int(230 * fade), int(200 * fade))
            for segs in b['bolts']:
                for p0, p1 in segs:
                    cv2.line(glow, p0, p1, glow_color, 6, cv2.LINE_AA)
                    cv2.line(sharp, p0, p1, sharp_color, 2, cv2.LINE_AA)
            alive.append(b)
    lightning_bolts = alive


def spawn_web_shot(origin, direction, scale):
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


def draw_web(glow, sharp, cx, cy, radius, fade, spokes=8, rings=4):
    if radius < 2:
        return
    base = int(245 * fade)
    color_glow = (base, base, base)
    csharp = min(int(255 * fade), 255)
    color_sharp = (csharp, csharp, csharp)
    for i in range(spokes):
        a = i * (2 * math.pi / spokes)
        x1 = int(cx + math.cos(a) * radius)
        y1 = int(cy + math.sin(a) * radius)
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


def draw_mystic_circle(glow, sharp, cx, cy, scale, t):
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


def draw_portal_warp(frame, cx, cy, base_r, t, fade=1.0):
    h, w = frame.shape[:2]
    r = int(base_r)
    r = min(r, cx, w - cx, cy, h - cy)
    if r < 15:
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
    tint[:] = (30, 80, 255)
    warped = cv2.addWeighted(warped, 0.75, tint, 0.25, 0)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1, cv2.LINE_AA)
    mask3 = cv2.merge([mask, mask, mask])

    roi = frame[y0:y0 + size, x0:x0 + size]
    composite = cv2.addWeighted(warped, fade, roi, 1 - fade, 0)
    blended = np.where(mask3 > 0, composite, roi)
    frame[y0:y0 + size, x0:x0 + size] = blended


def draw_portal_flame_ring(glow, sharp, cx, cy, r, fade=1.0):
    cv2.circle(glow, (cx, cy), r, (int(40 * fade), int(160 * fade), int(255 * fade)), 14, cv2.LINE_AA)
    cv2.circle(glow, (cx, cy), r, (int(80 * fade), int(220 * fade), int(255 * fade)), 6, cv2.LINE_AA)
    cv2.circle(sharp, (cx, cy), r, (int(160 * fade), int(240 * fade), int(255 * fade)), 3, cv2.LINE_AA)

    n_strands = 140
    for i in range(n_strands):
        a = i * (2 * math.pi / n_strands) + random.uniform(-0.02, 0.02)
        lick = random.uniform(4, 30)
        inner_x = cx + math.cos(a) * (r - 5)
        inner_y = cy + math.sin(a) * (r - 5)
        outer_x = cx + math.cos(a) * (r + lick)
        outer_y = cy + math.sin(a) * (r + lick)
        long_lick = lick > 18
        glow_color = (int(20 * fade), int(110 * fade), int(255 * fade)) if long_lick else (int(50 * fade), int(190 * fade), int(255 * fade))
        sharp_color = (int(30 * fade), int(150 * fade), int(255 * fade)) if long_lick else (int(120 * fade), int(220 * fade), int(255 * fade))
        cv2.line(glow, (int(inner_x), int(inner_y)), (int(outer_x), int(outer_y)), glow_color, 3, cv2.LINE_AA)
        cv2.line(sharp, (int(inner_x), int(inner_y)), (int(outer_x), int(outer_y)), sharp_color, 1, cv2.LINE_AA)


def draw_portal_sparks(glow, sharp, cx, cy, r, t, fade=1.0):
    n = 70
    for i in range(n):
        a = random.uniform(0, 2 * math.pi)
        radial = r + random.uniform(-4, 46)
        px = int(cx + math.cos(a) * radial)
        py = int(cy + math.sin(a) * radial)
        size = random.choice([1, 1, 2, 2, 3])
        warm = random.random() < 0.7
        glow_color = (int(30 * fade), int(140 * fade), int(255 * fade)) if warm else (int(70 * fade), int(200 * fade), int(255 * fade))
        sharp_color = (int(110 * fade), int(190 * fade), int(255 * fade)) if warm else (int(160 * fade), int(230 * fade), int(255 * fade))
        cv2.circle(glow, (px, py), size + 3, glow_color, -1, cv2.LINE_AA)
        cv2.circle(sharp, (px, py), size, sharp_color, -1, cv2.LINE_AA)


def spawn_portal(cx, cy, radius_target):
    portals.append({'x': cx, 'y': cy, 'max_r': radius_target, 'age': 0.0})


def update_and_draw_portals(frame, glow, sharp, dt):
    global portals
    alive = []
    open_dur = 0.45
    hold = 3.0
    fade_dur = 1.0
    total = open_dur + hold + fade_dur
    for p in portals:
        p['age'] += dt
        age = p['age']
        if age < open_dur:
            r = p['max_r'] * (age / open_dur)
            fade = 1.0
        elif age < open_dur + hold:
            r = p['max_r']
            fade = 1.0
        elif age < total:
            r = p['max_r']
            fade = 1.0 - (age - open_dur - hold) / fade_dur
        else:
            continue
        if r > 15:
            draw_portal_warp(frame, int(p['x']), int(p['y']), int(r), ring_time, fade)
            draw_portal_flame_ring(glow, sharp, int(p['x']), int(p['y']), int(r), fade)
            draw_portal_sparks(glow, sharp, int(p['x']), int(p['y']), int(r), ring_time, fade)
        alive.append(p)
    portals = alive


prev_time = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
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

    for label in list(circle_cooldown.keys()):
        circle_cooldown[label] = max(0.0, circle_cooldown[label] - dt)

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
                draw_mystic_circle(glow, sharp, palm[0], palm[1], scale, ring_time)
                active_gestures.append('circle')

            if is_fist(pts):
                angle = fist_angle_deg(pts)
                if near_diagonal(angle):
                    draw_claws(glow, sharp, pts, angle)
                    active_gestures.append('claws')

            spiderman_now = is_spiderman(pts)
            if spiderman_now and not spiderman_prev.get(label, False):
                origin = pts[0]
                d = dist(pts[8], pts[0]) + 1e-6
                direction = ((pts[8][0] - pts[0][0]) / d, (pts[8][1] - pts[0][1]) / d)
                spawn_web_shot(origin, direction, scale)
                active_gestures.append('web-shot')
            spiderman_prev[label] = spiderman_now

            if pinch_now and not pinch_prev.get(label, False):
                spawn_lightning(pinch_point[0], pinch_point[1])
                active_gestures.append('spark')
            pinch_prev[label] = pinch_now

            if is_draw_pose(pts):
                tip = ((pts[8][0] + pts[12][0]) // 2, (pts[8][1] + pts[12][1]) // 2)
                buf = circle_buffer.setdefault(label, [])
                buf.append((tip[0], tip[1], now))
                while buf and now - buf[0][2] > 1.6:
                    buf.pop(0)
                if len(buf) > 8 and circle_cooldown.get(label, 0.0) <= 0.0:
                    cx = sum(p[0] for p in buf) / len(buf)
                    cy = sum(p[1] for p in buf) / len(buf)
                    spread = max(dist((p[0], p[1]), (cx, cy)) for p in buf)
                    if spread > scale * 0.6:
                        total_angle = 0.0
                        prev_angle = math.atan2(buf[0][1] - cy, buf[0][0] - cx)
                        for p in buf[1:]:
                            a = math.atan2(p[1] - cy, p[0] - cx)
                            d = a - prev_angle
                            while d > math.pi:
                                d -= 2 * math.pi
                            while d < -math.pi:
                                d += 2 * math.pi
                            total_angle += d
                            prev_angle = a
                        if abs(total_angle) > math.radians(300):
                            spawn_portal(cx, cy, max(spread * 1.3, scale * 1.8))
                            circle_cooldown[label] = 2.0
                            buf.clear()
                            active_gestures.append('portal-open')
            else:
                circle_buffer[label] = []

            hands_data.append({'label': label, 'pinch': pinch_now, 'pinch_point': pinch_point})

    if len(hands_data) == 2 and hands_data[0]['pinch'] and hands_data[1]['pinch']:
        draw_tether(glow, sharp, hands_data[0]['pinch_point'], hands_data[1]['pinch_point'], ring_time)
        active_gestures.append('tether')

    update_and_draw_lightning(glow, sharp, dt)
    update_and_draw_web_shots(glow, sharp, dt)
    update_and_draw_portals(frame, glow, sharp, dt)

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
