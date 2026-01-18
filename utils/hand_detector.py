import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions


class HandDetector:
    """
    Hand detection and landmark tracking using MediaPipe Hands.
    Compatible with MediaPipe >= 0.10.30 (Windows-safe).
    """

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # MediaPipe modules (FIXED IMPORT)
        self.mp_hands = solutions.hands
        self.mp_draw = solutions.drawing_utils

        # Hand detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

        self.results = None

    # --------------------------------------------------------
    # Detect hands and optionally draw landmarks
    # --------------------------------------------------------
    def find_hands(self, img, draw: bool = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        return img

    # --------------------------------------------------------
    # Get all landmark positions for a hand
    # --------------------------------------------------------
    def find_positions(self, img, hand_index: int = 0, draw: bool = True):
        landmarks = []

        if self.results and self.results.multi_hand_landmarks:
            if hand_index >= len(self.results.multi_hand_landmarks):
                return landmarks

            hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = img.shape

            for idx, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((idx, cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

        return landmarks

    # --------------------------------------------------------
    # Get a single fingertip position (default: index finger)
    # --------------------------------------------------------
    def get_finger_tip(self, img, finger_id: int = 8, hand_index: int = 0):
        landmarks = self.find_positions(img, hand_index, draw=False)

        for idx, x, y in landmarks:
            if idx == finger_id:
                return x, y

        return None, None

    # --------------------------------------------------------
    # Check if a finger is raised (tip above PIP joint)
    # --------------------------------------------------------
    def is_finger_up(
        self,
        img,
        finger_tip_id: int,
        finger_pip_id: int,
        hand_index: int = 0,
    ) -> bool:
        landmarks = self.find_positions(img, hand_index, draw=False)

        lm_dict = {idx: (x, y) for idx, x, y in landmarks}

        if finger_tip_id in lm_dict and finger_pip_id in lm_dict:
            # Y-axis: smaller value = higher on screen
            return lm_dict[finger_tip_id][1] < lm_dict[finger_pip_id][1]

        return False

    # --------------------------------------------------------
    # Cleanup (good practice)
    # --------------------------------------------------------
    def close(self):
        self.hands.close()
