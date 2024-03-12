import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
import numpy as np
#image processing model

image1 = 'vectorized\image6.jpg'
image2 = 'vectorized\image5.jpg'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def image_encoder(img):
    #print(Image.fromarray(img))
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def cosine_similarity_score(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = image_encoder(test_img)
    img2 = image_encoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

def euclidean_distance(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = image_encoder(test_img)
    img2 = image_encoder(data_img)

    # Flatten the vectors before calculating Euclidean distance
    vec1 = img1.view(-1).cpu().detach().numpy()
    vec2 = img2.view(-1).cpu().detach().numpy()

    score = np.linalg.norm(vec1 - vec2)
    
    return score

def print_values(image1, image2):
    print(f"Similarity Score (Cosine Similarity): ", round(cosine_similarity_score(image1, image2), 2))
    print(f"Euclidean Distance: ", round(euclidean_distance(image1, image2), 2))

if __name__ == "__main__":
    print_values(image1, image2)
