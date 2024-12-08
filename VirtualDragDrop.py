import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect:
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if (
            cx - w // 2 < cursor[0] < cx + w // 2
            and cy - h // 2 < cursor[1] < cy + h // 2
        ):
            self.posCenter = cursor[0], cursor[1]


rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True, flipType=False)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        center1 = hand1["center"]
        handType1 = hand1["type"]

        length, info, img = detector.findDistance(
            lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255)
        )
        if length < 40:
            cursor = lmList1[8]
            print(length)
            for rect in rectList:
                rect.update(cursor)

        if len(hands) == 2:

            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2["center"]
            handType2 = hand2["type"]

            length, info, img = detector.findDistance(
                lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0)
            )

    ## Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size

        cv2.rectangle(
            imgNew,
            (cx - w // 2, cy - h // 2),
            (cx + w // 2, cy + h // 2),
            colorR,
            cv2.FILLED,
        )
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)
