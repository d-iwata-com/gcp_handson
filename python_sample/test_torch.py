# パッケージのimport
import numpy as np
import json
from PIL import Image
import glob


import torch
import torchvision
from torchvision import models, transforms

# VGG-16モデルのインスタンスを生成
net = models.vgg16(pretrained=True)
net.eval()  # 推論モードに設定

#  画像読み込み
files = glob.glob("./data/*.jpg")

# ILSVRCのラベル情報をロードし辞書型変数を生成します
class_index = json.load(open('./imagenet_class_index.json', 'r'))

for file in files:
    print(file)
    img = Image.open(file)  # [高さ][幅][色RGB]
    # 前処理の後、バッチサイズの次元を追加する
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(resize),  # 短い辺の長さがresizeの大きさになる
        transforms.CenterCrop(resize),  # 画像中央をresize × resizeで切り取り
        transforms.ToTensor(),  # Torchテンソルに変換
        transforms.Normalize(mean, std)  # 色情報の標準化
        ])
    img_transformed = transform(img)  # torch.Size([3, 224, 224])
    inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])

    # モデルに入力し、モデル出力をラベルに変換する
    out = net(inputs)  # torch.Size([1, 1000])
    maxid = np.argmax(out.detach().numpy())

    result = class_index[str(maxid)][1]

    # 予測結果を出力する
    print("入力画像の予測結果：", result)


