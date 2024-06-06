from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans

# パージングモジュール
model = YOLO("runs/train/weights/best3.pt")
img = cv2.imread("images/path_to_image.jpg")
results = model(img)

# オブジェクトの種類を調べる
for e in results[0].boxes.cls.cpu():
    print(e, model.names[int(e)])

# Pantsの領域を抽出
pants_mask = None
for mask, cls in zip(results[0].masks.xy, results[0].boxes.cls):
    if model.names[int(cls)] == "pants":
        pants_mask = mask
        break

if pants_mask is None:
    raise ValueError("Pants not found in the image.")

# マスクを二値化
mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask_img, [np.array(pants_mask, dtype=np.int32)], 255)

# 主要色決定モジュール
def get_dominant_color(image, mask, k=3):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    masked_lab_image = lab_image[mask == 255]
    clt = KMeans(n_clusters=k)
    clt.fit(masked_lab_image)
    return clt.cluster_centers_

dominant_colors = get_dominant_color(img, mask_img)

# 調和評価モジュール
def calculate_harmony(color):
    L, a, b = color
    V_star = L
    C_star = np.sqrt(a**2 + b**2)
    H_star = np.arccos(a / np.sqrt(a**2 + b**2)) if b >= 0 else -np.arccos(a / np.sqrt(a**2 + b**2))
    return V_star, C_star, H_star

V_star, C_star, H_star = calculate_harmony(dominant_colors[0])

# 色推奨モジュール
def recommend_colors(dominant_color, harmony_criteria=None):
    L, a, b = dominant_color
    recommended_color = (L, -a, -b)  # 例: コンプリメンタリーカラー
    return recommended_color

new_color = recommend_colors(dominant_colors[0])

# 色変換モジュール
def apply_color_transformation(image, mask, new_color):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_image)
    
    # 新しい色のaとbを適用
    a[mask == 255] = new_color[1]
    b[mask == 255] = new_color[2]
    
    # 変換後のLab画像を合成
    transformed_lab_image = cv2.merge([L, a, b])
    return cv2.cvtColor(transformed_lab_image, cv2.COLOR_LAB2BGR)

# パンツの色を変換
transformed_img = apply_color_transformation(img, mask_img, new_color)
cv2.imwrite("output/change_pants_color.jpg", transformed_img)
