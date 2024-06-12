import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import japanize_matplotlib

start_time = time.time()

# モデルの読み込み
model = YOLO("runs/train/weights/best3.pt")

# 画像の読み込み
img = cv2.imread("images/path_to_image12.jpg")

# 推論の実行
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

    # 対象領域を128x128にリサイズ
    #img_resized = cv2.resize(seg_img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # 透明部分を取り除く
    #non_zero_pixels = img_resized[img_resized.sum(axis=2) != 0]
    
    # 透明部分を取り除く(リサイズしない場合)
    non_zero_pixels = seg_img[seg_img.sum(axis=2) != 0][:, :3]

    # リサイズされた画像を表示
    """plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("(128x128)")
    plt.axis("off")
    plt.show()"""

    if non_zero_pixels.size == 0:
        print("透明部分以外のピクセルがありませんでした。")
    else:
        # K-meansクラスタリングを適用
        k = 5  # クラスタの数
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(non_zero_pixels)

        # クラスタの中心を取得
        centers = kmeans.cluster_centers_

        # 各クラスタのラベルを取得
        labels = kmeans.labels_

        # 各クラスタのサイズを計算
        label_counts = np.bincount(labels)

        # 最大クラスタのインデックスを取得
        dominant_color_index = np.argmax(label_counts)

        # 最大クラスタのRGB値を取得
        dominant_color = centers[dominant_color_index]
        
        end_time = time.time()
        print(f"処理時間: {end_time - start_time} 秒")

        # 結果を表示
        print(f"主要色 (RGB): {dominant_color}")

        # 主要色をプレビュー
        dominant_color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        dominant_color_img[:] = dominant_color

        plt.imshow(cv2.cvtColor(dominant_color_img, cv2.COLOR_BGR2RGB))
        plt.title("主要色")
        plt.axis("off")
        plt.show()
