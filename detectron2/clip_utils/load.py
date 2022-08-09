import itertools
import clip
import torch

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant", "sofa", "tvmonitor"
]

VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

T2_CLASS_NAMES = [
    "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "microwave", "oven", "toaster", "sink", "refrigerator"
]

T3_CLASS_NAMES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]

T4_CLASS_NAMES = [
    "bed", "toilet", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl"
]

UNK_CLASS = ["object"]

# 实例的类别的名称
VOC_COCO_CLASS_NAMES = tuple(
    itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


class ClipProcess:
    def __init__(self, class_names=VOC_COCO_CLASS_NAMES, name="/lct/clip/OWOD-master/ViT-B-32.pt"):
        self.class_names = class_names
        self.name = name

        self.cuda_clip_model = None
        self.cuda_clip_preprocess = None
        self.cuda_text_features = None
        self.cuda_text_token = None

        self.cpu_clip_model = None
        self.cpu_clip_preprocess = None
        self.cpu_text_features = None
        self.cpu_text_token = None

        self._load()
        self.init()

    def _load(self):
        self.cuda_clip_model, self.cuda_clip_preprocess = clip.load(self.name, device='cuda')
        self.cuda_clip_model, self.cuda_clip_preprocess = clip.load(self.name, device='cpu')

    def init(self):
        text_token = clip.tokenize(self.class_names)
        with torch.no_grad():
            self.cuda_text_features = self.cuda_clip_model.encode_text(text_token)
            self.cpu_text_features = self.cuda_clip_model.encode_text(text_token)

    def get_text_features(self, device):
        if device == 'cuda':
            return self.cuda_text_features
        else:
            return self.cpu_text_features
