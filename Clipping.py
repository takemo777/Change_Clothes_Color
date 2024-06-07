from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/train/weights/best3.pt")

img = cv2.imread("images/path_to_image18.jpg")

results = model(img)

# 検出対象のラベル
target_labels = ['coat', 'dress', 'jacket', 'shirt', 't-shirt']

# マスクの初期化
h, w = img.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)

# 対象オブジェクトのマスクを結合
found_target = False  # 対象オブジェクトが検出されたかどうか
for seg, cls in zip(results[0].masks.data.cpu(), results[0].boxes.cls.cpu()):
    label = model.names[int(cls)]
    if label in target_labels:
        found_target = True
        seg_resized = cv2.resize(seg.numpy().astype(np.uint8) * 255, (w, h))
        mask = cv2.bitwise_or(mask, seg_resized)

# 対象オブジェクトが検出されなかった場合
if not found_target:
    print("指定されたラベルのオブジェクトが検出されませんでした。")
else:
    # マスクの外側の境界を取得
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        mask_cropped = mask[y:y+h, x:x+w]
        img_cropped = img[y:y+h, x:x+w]
    else:
        mask_cropped = mask
        img_cropped = img

    # 対象領域を抽出し、背景を透明にする
    seg_img = cv2.bitwise_and(img_cropped, img_cropped, mask=mask_cropped)

    # アルファチャネルを追加
    b, g, r = cv2.split(seg_img)
    alpha = mask_cropped

    # 透明背景の画像を作成
    transparent_img = cv2.merge([b, g, r, alpha])

    # 結果の保存
    output_path = "tops_area.png"
    cv2.imwrite(output_path, transparent_img)
