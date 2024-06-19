from ultralytics import YOLO
import os
import torch

# Diretório do dataset
dataset_path = 'dataset'

# Verificar se o treinamento será na CPU
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

# Criação do modelo
model = YOLO('yolov8n-cls.yaml')

# Diretório para salvar o melhor modelo
best_model_path = 'best_model.pt'

# Treinamento
model.train(data=dataset_path, 
            epochs=100, 
            imgsz=224, 
            batch=16, 
            device=device,
            save=True,  # Salvar modelos durante o treinamento
            save_dir='runs/train'  # Diretório para salvar os modelos
           )

# Encontrar o modelo com melhor desempenho (menor valor de loss)
best_model_file = os.path.join('runs', 'train', 'weights', 'best.pt')
if os.path.exists(best_model_file):
    os.rename(best_model_file, best_model_path)

# Validação
metrics = model.val(data=dataset_path)

# Exibir os resultados
print(metrics)
