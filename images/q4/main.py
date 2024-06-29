import os
import PIL

import torch, torchvision
from typing import Callable, List, Union
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Checar se este processo est[a de acordo com o paper sugerido

# 1- Adicionar input redimensionando a imagem passar pelo modelo
# 2- Visualizar as features de saída
# 3- Adicionar segunda imagem e visualizar as features de saída
# 4- Calcular a distância entre as features das duas imagens
# 5- Visualizar as correspondências entre as features das duas imagens
# 6- Manipular campo receptivo para melhor manipular o input


# Get current file path
file_path = os.path.abspath(__file__)



# Função auxiliar que controi um forward hook que coloca na lista informada o tensor com features
# produzidas por um torch.nn.Module.
def append_output_features_to(some_list: List[torch.Tensor]) -> Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], Union[None, torch.Tensor]]:
    def hook(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> Union[None, torch.Tensor]:
        some_list.append(output)
    return hook

# x and y here denotes points in the feature space of the two images. x image 1 and y image 2
def compute_distance(D1x, D1y, D2x, D2y, D3x, D3y):
    """
    Compute the weighted sum of Euclidean distances between feature descriptors.
    But the dimensions are different, 256 + 512 + 512.
    To solve this, we normalize the features to unit variance.

    """
    d1 = np.sqrt(cdist(D1x, D1y, 'euclidean')) # Distance in F1
    d2 = cdist(D2x, D2y, 'euclidean')           # Distance in F2
    d3 = cdist(D3x, D3y, 'euclidean')           # Distance in F3
    return np.sqrt(2) * d1 + d2 + d3

def match_features(D1_img1, D2_img1, D3_img1, D1_img2, D2_img2, D3_img2, threshold):
    """
    Match feature points between two images based on the defined conditions.
    """
    distances = compute_distance(D1_img1, D1_img2, D2_img1, D2_img2, D3_img1, D3_img2)
    matches = []
    for i, distance_row in enumerate(distances):
        min_distance = np.min(distance_row)
        min_index = np.argmin(distance_row)
        # Check if there's a distance smaller than threshold * min_distance
        if not np.any(distance_row < threshold * min_distance):
            matches.append((i, min_index))
    return matches

# Método principal
def main():
    # Loaded vgg16 pretrained model
    model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
    
    # Imprimir a arquitetura da rede na saída padrão só por curiosidade.
    print(model)
    print()

    if torch.cuda.is_available():
        model = model.cuda()


    # Load input images
    input_image1 = PIL.Image.open(file_path.replace('main.py', 'hp_calvo1.jpg'))
    input_image2 = PIL.Image.open(file_path.replace('main.py', 'hp_calvo2.jpg'))

    # input_image1 = PIL.Image.open(file_path.replace('main.py', 'Questionario-4-Bricks1.jpg'))
    # input_image2 = PIL.Image.open(file_path.replace('main.py', 'Questionario-4-Bricks2.jpg'))

    # input_image1 = PIL.Image.open(file_path.replace('main.py', 'Questionario-4-Building1.jpg'))
    # input_image2 = PIL.Image.open(file_path.replace('main.py', 'Questionario-4-Building2.jpg'))


    # Preprocess image to be compatible with the model
    # Checar se este processo esta de acordo com o paper sugerido
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor1 = preprocess(input_image1)
    input_tensor2 = preprocess(input_image2)
    
    # Só temos uma imagem, que é um tensor de shape (3, 224, 224), enquanto que a rede espera um
    # tensor de shape (N, 3, 224, 224), onde N é a quantidade de imagens no batch. Logo, é preciso
    # adicionar uma dimensão a mais no nosso tensor.
    input_batch1 = input_tensor1.unsqueeze(0)
    input_batch2 = input_tensor2.unsqueeze(0)


    input_batch1 = input_batch1.cuda()
    input_batch2 = input_batch2.cuda()

    # Instrumentar o módulo para capturar features recebidas e produzidas por submódulos.
    computed_features1_pool3 = [] # Pool3 1, 256, 28, 28
    computed_features2_pool3 = [] # Pool3 1, 256, 28, 28
    computed_features1_pool4 = []     # Pool4 14x14x512
    computed_features2_pool4 = []     # Pool4 14x14x512
    computed_features1_pool5 = []     # Pool5 7x7x512
    computed_features2_pool5 = []     # Pool5 7x7x512


    #Features F1 = Pool3 Kronicker product Image 2x2x1


    # The pool5 layer is not used for feature because it is affected by 
    # specific classification objects thus not suitable for detecting
    # general features.

    # 16th -> Pool3  Output -> 28x28x256 = F1
    model.features[16].register_forward_hook(append_output_features_to(computed_features1_pool3))
    # 24th -> Pool4  Output -> 14x14x512 kronicker i2x2x1 = F2
    model.features[23].register_forward_hook(append_output_features_to(computed_features1_pool4))
    #print the size of each dimension of computed_features1
    # 31th -> Pool5  Output -> 7x7x512 krnicker i4x4x1 = F3
    model.features[30].register_forward_hook(append_output_features_to(computed_features1_pool5))

    # Aplicar o modelo de classificação sobre a imagem de entrada. Como não estamos em tempo de
    # treinamento então não é preciso calcular o gradiente para aplicar backpropagation
    # posteriormente.

    # Map each feature to the corresponding space in the image. 
    # Pool3 comprehends 8x8 receptive field, Pool4 16x16
    # I want to map the features to the image space to visualize them, where each feature above
    # a threshold will be represented by a dot on the center of the receptive field.

    with torch.no_grad():
        output = model(input_batch1)
    
    img1f1 = computed_features1_pool3[0].cpu().numpy() 

    identity2x2x1 = np.array([[1, 1], [1, 1]])
    identity4x4x1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    # f2 = computed_features1_pool4[0] kronicker identity2x2x1
    img1f2 = np.kron(computed_features1_pool4[0].cpu().numpy(), identity2x2x1)

    # f3 = computed_features1_pool5[0] kronicker identity4x4x1
    img1f3 = np.kron(computed_features1_pool5[0].cpu().numpy(), identity4x4x1)
    print(img1f1.shape, img1f2.shape, img1f3.shape)

    # normalize to unit variance f1 = f1/standard deviation(f1)
    img1f1 = img1f1/np.std(img1f1)
    img1f2 = img1f2/np.std(img1f2)
    img1f3 = img1f3/np.std(img1f3)

    # Image 2
    with torch.no_grad():
        output = model(input_batch2)
    
    img2f1 = computed_features1_pool3[0].cpu().numpy()
    img2f2 = np.kron(computed_features1_pool4[0].cpu().numpy(), identity2x2x1)
    img2f3 = np.kron(computed_features1_pool5[0].cpu().numpy(), identity4x4x1)
    print(img2f1.shape, img2f2.shape, img2f3.shape)
    img2f1 = img2f1/np.std(img2f1)
    img2f2 = img2f2/np.std(img2f2)
    img2f3 = img2f3/np.std(img2f3)

    #* Calculate the distance between the features of the two images using Euclidian distance
    # d(x,y) = sqrt(2*d1(x,y) + d2(x,y) + d3(x,y))
    # The distance computed with pool3 descriptors d1(x, y) is compensated with a weight √2 because D1 is 256-d 
    # whereas D2 and D3 are 512-d.
    #
    # Feature point x is matched to y if the following conditions are satisfied
    # d(x, y) is the smallest of all d(·, y)
    # There does not exist a d(z, y) such that d(z, y) < θ ·
    # d(x, y). θ is a parameter valued greater than 1 and is
    # called the matching threshold.

    # Calculate the distance between the features of the two images using Euclidian distance
    # to match the features of the two images 
    # if the distance is smaller than the threshold then the features are matched
    # Also visualize the matching features



    threshold = 1.0
    D1_img1 = img1f1.reshape(-1, img1f1.shape[-1])
    D2_img1 = img1f2.reshape(-1, img1f2.shape[-1])
    D3_img1 = img1f3.reshape(-1, img1f3.shape[-1])

    D1_img2 = img2f1.reshape(-1, img2f1.shape[-1])
    D2_img2 = img2f2.reshape(-1, img2f2.shape[-1])
    D3_img2 = img2f3.reshape(-1, img2f3.shape[-1])


    matches = match_features(D1_img1, D2_img1, D3_img1, D1_img2, D2_img2, D3_img2, threshold)
    print(f'Found {len(matches)} matches')


    # Draw both images side by side glued together




# Normalizar e mostrar os scores de confiança para cada uma das 1000 classes
def model_output(output):
    # Carregar o nome das classes.
    with open(file_path.replace('main.py', 'imagenet_classes.txt')) as file:
        classes = [s.strip() for s in file.readlines()]
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    for c, p in zip(classes, probabilities):
        print(f'{100*p:1.2f}%: {c}')

if __name__ == '__main__':
    main()
