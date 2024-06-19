from ultralytics import YOLO
import torch
from PIL import Image

# Caminho para o modelo salvo
best_model_path = 'runs/classify/train/weights/best.pt'

# Carregar o modelo salvo
model = YOLO(best_model_path)

# Função para fazer inferência em uma imagem
def infer_image(image_path):
    image = Image.open(image_path)
    results = model(image)
    return results

# ARAUCARIA
# image_path = 'dataset/test/araucaria/Captura de tela de 2024-06-19 15-59-01.png'

# CAPIVARA
# image_path = 'dataset/test/capivara/Captura de tela de 2024-06-19 15-54-03.png'

# GRALHA AZUL
# image_path = 'dataset/test/gralha_azul/Captura de tela de 2024-05-25 18-30-09gralhaazul.png'

# PINHÃO
image_path = 'dataset/test/pinhao/Captura de tela de 2024-06-19 15-35-23.png'

results = infer_image(image_path)

# Exibir os resultados da inferência
for result in results:
    result.show()  # Para exibir a imagem com as detecções
    print(result)  # Para exibir as informações das detecções
