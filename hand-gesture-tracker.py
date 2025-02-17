import cv2
import mediapipe as mp
import time

class HandGestureTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def get_hand_position(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return lmList

def count_fingers_up(lmList, isLeftHand):
    fingers = []
    if isLeftHand:
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    for id in range(8, 21, 4):
        if lmList[id][2] < lmList[id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def identify_gesture(fingers):
    if fingers == [0, 1, 0, 0, 0]:
        return "Pointing"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Five"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Victory"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [0, 0, 1, 0, 0]:
        return "Middle Finger"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 1, 0, 0, 1]:
        return "Rock"
    elif fingers == [1, 0, 0, 0, 1]:
        return "Call Me"
    elif fingers == [1, 1, 1, 0, 0]:
        return "Three"
    elif fingers == [1, 0, 1, 0, 1]:
        return "OK"
    elif fingers == [0, 1, 0, 0, 1]:
        return "Peace"
    elif fingers == [1, 1, 0, 0, 0]:
        return "Gun"
    elif fingers == [1, 1, 1, 1, 1] and max(fingers) - min(fingers) < 1:
        return "Claw"
    elif fingers == [1, 1, 1, 0, 1]:
        return "Spider-Man"
    elif fingers == [0, 0, 0, 0, 1]:
        return "Thumbs Down"
    else:
        return None

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = HandGestureTracker()

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            continue

        img = detector.detect_hands(img)
        hand_positions = []

        for handNo in range(2):
            lmList = detector.get_hand_position(img, handNo=handNo, draw=False)

            if len(lmList) != 0:
                hand_center_x = sum([lm[1] for lm in lmList]) // len(lmList)
                hand_positions.append((handNo, hand_center_x, lmList))

        hand_positions.sort(key=lambda x: x[1])

        for handNo, _, lmList in hand_positions:
            isLeftHand = lmList[17][1] > lmList[5][1]
            position_x = 10 if _ < img.shape[1] // 2 else img.shape[1] - 300

            fingers = count_fingers_up(lmList, isLeftHand)
            totalFingers = sum(fingers)

            cv2.putText(img, f'H{handNo+1}: {totalFingers}', (position_x, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

            finger_messages = {
                0: "Thumb",
                1: "Index",
                2: "Middle",
                3: "Ring",
                4: "Little"
            }

            for i, (finger, message) in enumerate(zip(fingers, finger_messages.values())):
                if finger == 1:
                    cv2.putText(img, message, (position_x, 150 + i * 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

            gesture = identify_gesture(fingers)
            if gesture:
                cv2.putText(img, f"Gesture: {gesture}", (position_x, 400), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Hand Tracker", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    