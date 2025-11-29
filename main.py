import cv2
import mediapipe as mp
import numpy as np

# --- Настройки камеры ---
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10, 150)

# --- Mediapipe ---
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
draw = mp.solutions.drawing_utils

mpFace = mp.solutions.face_mesh
face_mesh = mpFace.FaceMesh(max_num_faces=1, refine_landmarks=True)

pTime = 0
overlay_size = 300

# --- Функции ---
def get_fingers_status(handLms):
    fingers = []
    fingers.append(1 if handLms.landmark[4].x < handLms.landmark[3].x else 0)
    for tip, pip in zip([8,12,16,20],[6,10,14,18]):
        fingers.append(1 if handLms.landmark[tip].y < handLms.landmark[pip].y else 0)
    return fingers

def middle(hand_status):
    return hand_status[2] == 1

def swag(hand_status):
    return hand_status == [1,0,0,0,1]

def up(hand_status):
    return hand_status == [0,1,0,0,0]

def hands_together(hand1, hand2, img_w, img_h, threshold=80):
    x1, y1 = hand1.landmark[0].x*img_w, hand1.landmark[0].y*img_h
    x2, y2 = hand2.landmark[0].x*img_w, hand2.landmark[0].y*img_h
    center_dist = np.hypot(x1-x2, y1-y2)
    finger_dist = np.mean([np.hypot(hand1.landmark[i].x*img_w - hand2.landmark[i].x*img_w,
                                    hand1.landmark[i].y*img_h - hand2.landmark[i].y*img_h)
                           for i in [8,12,16,20]])
    return center_dist < threshold and finger_dist < threshold

def mouth(hand_landmarks, mouth_box, img_w, img_h):
    finger_x = int(hand_landmarks.landmark[8].x*img_w)
    finger_y = int(hand_landmarks.landmark[8].y*img_h)
    mx1,my1,mx2,my2 = mouth_box
    return mx1 <= finger_x <= mx2 and my1 <= finger_y <= my2

def overlay_image(img, overlay, x, y):
    if overlay is None:
        return img
    overlay = cv2.resize(overlay, (overlay_size, overlay_size))
    h, w, _ = img.shape
    oh, ow, _ = overlay.shape
    if x + ow > w: ow = w - x; overlay = overlay[:, :ow]
    if y + oh > h: oh = h - y; overlay = overlay[:oh]
    img[y:y+oh, x:x+ow] = overlay
    return img

# --- Основной цикл ---
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    results_hands = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results_face = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Рот
    mouth_box = None
    if results_face.multi_face_landmarks:
        faceLms = results_face.multi_face_landmarks[0]
        upper_lip = faceLms.landmark[13]
        lower_lip = faceLms.landmark[14]
        ux, uy = int(upper_lip.x*w), int(upper_lip.y*h)
        lx, ly = int(lower_lip.x*w), int(lower_lip.y*h)
        size = 70
        mx1 = min(ux,lx)-size
        my1 = min(uy,ly)-size
        mx2 = max(ux,lx)+size
        my2 = max(uy,ly)+size
        mouth_box = (mx1,my1,mx2,my2)

    hands_list = results_hands.multi_hand_landmarks or []
    hands_status = [get_fingers_status(h) for h in hands_list]

    middle_finger_left = False
    middle_finger_right = False
    finger_near_mouth = False

    for i, handLms in enumerate(hands_list):
        status = hands_status[i]
        hand_label = results_hands.multi_handedness[i].classification[0].label

        # --- Отображение точек и соединений рук ---
        draw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # Средние пальцы
        if middle(status):
            if hand_label=='Left':
                middle_finger_left = True
            else:
                middle_finger_right = True

        # Swag
        if swag(status):
            overlay = cv2.imread("swag.jpg")
            img = overlay_image(img, overlay, x=50, y=100)

        # Указательный вверх
        if up(status):
            overlay = cv2.imread("up.jpg")
            img = overlay_image(img, overlay, x=50, y=100)

        # Палец у рта
        if mouth_box is not None and mouth(handLms, mouth_box, w, h):
            finger_near_mouth = True
            overlay = cv2.imread("mouth.jpg")
            img = overlay_image(img, overlay, x=50, y=100)

    # Две руки вместе
    if len(hands_list) == 2 and hands_together(hands_list[0], hands_list[1], w, h):
        overlay = cv2.imread("scare.jpg")
        img = overlay_image(img, overlay, x=50, y=100)

    # Средние пальцы обеих рук
    if middle_finger_left and middle_finger_right:
        overlay = cv2.imread("middle.jpg")
        img = overlay_image(img, overlay, x=50, y=100)

    cv2.imshow("Hand Gesture System", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
