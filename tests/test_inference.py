import torch
from facenet.models.mtcnn import MTCNN, RNet, ONet, PNet
from facenet.models.inception_resnet_v1 import InceptionResnetV1
from face_config import FaceRecognitionConfig
import os
import cv2

device = "cuda"

mtcnn = MTCNN(device=device)

input = cv2.imread("data/anh-son-tung-mtp-thumb.jpg")
faces = mtcnn(input)
print(faces.shape)

boxes, probs, points = mtcnn.detect(input, landmarks=True)
print(boxes.shape, probs.shape, points.shape)