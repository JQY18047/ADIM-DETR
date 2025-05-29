import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('')
    model.load(r'')  # loading pretrain weights  预训练权重
    model.train(data='../Aown/data.yaml',
                cache=False,
                imgsz=640,
                epochs=5,
                batch=4,
                workers=0,
                device='0',
                # resume='', # last.pt path
                project='../runs/train',
                name='exp',
                )