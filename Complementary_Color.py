from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/train/weights/best3.pt")

img = cv2.imread("images/path_to_image2.jpg")

results = model(img)

# 色相シフトを適用する関数
def shift_hue(image, shift_value):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0].astype(int) + shift_value) % 180
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# オブジェクトの種類と色相シフトを適用
for result in results:
    # 画像と同じサイズの空のマスクを作成
    mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for mask, cls in zip(result.masks.xy, result.boxes.cls.cpu().numpy()):
        # マスクのポリゴン座標を使ってマスクを描画
        cv2.fillPoly(mask_img, [np.array(mask, dtype=np.int32)], 1)
        
        object_name = model.names[int(cls)]
        
        # オブジェクト領域を抽出
        object_region = cv2.bitwise_and(img, img, mask=mask_img)
        
        shifted_region = shift_hue(object_region, 90)
        
        # 色相シフトされた領域を元の画像に適用
        img[mask_img == 1] = shifted_region[mask_img == 1]
        
        print(f"オブジェクト: {object_name}")
        
        # 次のオブジェクトのためにマスクをリセット
        mask_img.fill(0)

cv2.imwrite("output/complementary_color.jpg", img)
