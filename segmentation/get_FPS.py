from models import DeepLabV1
 
 
model = DeepLabV1("/root/autodl-tmp/nsrom11/segmentation/data/models/voc12/deeplabv2_resnet101_msc/train_our/checkpoint_final.pth")
model.predict(source="/root/autodl-tmp/nsrom11/segmentation/data/features/voc12/deeplabv2_resnet101_msc/test_our/logit",save=True,save_conf=True,save_txt=True,name='output')
 
#source��ΪҪԤ���ͼƬ���ݼ��ĵ�·��
#save=TrueΪ����Ԥ����
#save_conf=TrueΪ����������Ϣ
#save_txt=TrueΪ����txt���������yolov8����ͼƬ��Ԥ�ⲻ������ʱ��������txt�ļ�