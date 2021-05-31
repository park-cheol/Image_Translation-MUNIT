import glob
import random
import os
import numpy as np

from PIL import Image

import torch
import torch.utils.data
import torchvision.transforms as transforms


"""우리 데이터 형태와 맞지 않은 코드"""
# class ImageDataset(torch.utils.data.Dataset):
#
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.transforms = transforms.Compose(transforms_)
#
#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#         if mode == "train":
#             self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
#             # 왜 A/B로 도메인을 나누지 않고 한 변수에 extend 한 이유: 사진 자체가 한사진에 AB동시에 들어 있으므로
#
#     def __getitem__(self, index):
#
#         img = Image.open(self.files[index % len(self.files)])
#         w, h = img.size # img.size = (width, height)
#
#         # 왼쪽 오른쪽으로 나누는 이유:사진 자체가 한사진에 AB동시에 들어 있으므로
#         # PIL.crop경우 왼쪽상단 모서리를 기점으로(시작한점위치, 위치, w를 얼마나, h를 얼마나)
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))
#
#         if np.random.random() < 0.5:
#             # Image.fromarray: 배열을 PIL이미지 객체로 반환
#             # todo:원래 RGBA여서 "RGB"로 변경후 plt해보면 사진이 꺠짐 그대로 가능한지
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
#             # [::-1]: 그 배열을 reverse
#         img_A = self.transforms(img_A)
#         img_B = self.transforms(img_B)
#
#         return {"A": img_A, "B": img_B}
#
#     def __len__(self):
#         return len(self.files)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))
        """glob: 파일들의 리시트를 뽑을 때 사용, 경로명을 이용해서 가져올 수 있다.
        glob.glob('*.exe') # 현재 디렉토리에서 모든 .exe파일
        glob.glob('file?.*') # file1.txt file1.exe file2.txt .....
        glob.glob(r'data/U*') # data파일에서 u시작하는 파일
        """
    """슬라이싱을 구현할 수 있도록 도우며 리스트에서 슬라이싱을 하게 되면 내부적으로 이 메소드를 실행
    객체에서도 슬라이싱을 하기위해서는 이 메소드가 필수적
    원래라면 a = class()
    a.list[2:5] 해야함
    def __getitem__(self, idx): return self.list[idx] 해주면됌
    a[2:5]
    """

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned: # unpaired
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # grayscle 에서 rgb로 전환 (아니면 .convert('RGB')도 가능)
        if image_A.mode != "RGB": # 즉 image_A가 흑백이라면
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        # 위에서 설정한 전처리 실행
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self): # len 실행 시 실제로 __len__메서드 호출
        return max(len(self.files_A), len(self.files_B))

    # A와B 파일을 따로따로 return
def to_rgb(image):
    # Image.new(mode, szie, color) : 새로운 이미지 생성
    rgb_image = Image.new("RGB", image.size)
    # .paste(추가할 이미지, 붙일 위치(가로, 세로)): 이미지붙이기
    rgb_image.paste(image)
    return rgb_image









