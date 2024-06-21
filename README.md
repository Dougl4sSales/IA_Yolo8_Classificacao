# Modelo de Classificação com YOLOv8

## Funcionalidade do Modelo
### Detecção de Objetos
O YOLOv8  é uma rede neural convolucional (CNN) que realiza a detecção de objetos em tempo real. Ele funciona dividindo a imagem de entrada em uma grade e, para cada célula da grade, prevê uma série de caixas delimitadoras (bounding boxes) e as probabilidades de classe para cada uma dessas caixas.

### Classificação de Objetos
Uma vez que os objetos são detectados, o modelo classifica cada objeto em uma das categorias definidas. No caso deste projeto, as categorias são:

* Araucárias: árvores características da região sul do Brasil.
* Capivaras: o maior roedor do mundo, encontrado principalmente na América do Sul.
* Gralha Azul: um pássaro símbolo do estado do Paraná.
* Pinhão: semente da Araucária, consumida como alimento.

### Arquitetura e Treinamento
O YOLOv8 utiliza uma arquitetura de rede neural profunda que é otimizada para velocidade e precisão. Durante o treinamento, o modelo aprende a identificar e classificar objetos com base em um grande conjunto de dados rotulados, que contém exemplos de imagens de cada categoria. O treinamento envolve a minimização de uma função de perda que penaliza predições incorretas de caixas delimitadoras e classes.

### Objetivo do Modelo
O objetivo principal do modelo YOLOv8, neste contexto, é facilitar a identificação automática de Araucárias, Capivaras, Gralha Azul e Pinhão em imagens. Isto pode ser útil em diversas aplicações, tais como:

Monitoramento Ambiental: Auxiliar na preservação de espécies e no monitoramento de ecossistemas.
Agricultura: Identificação de áreas com Araucárias para a colheita sustentável de pinhão.
Pesquisa Científica: Coleta de dados para estudos ecológicos e de biodiversidade.
Educação Ambiental: Ferramenta educativa para ensinar sobre a fauna e flora locais.

## Requisitos
Para seguir este tutorial, você precisará dos seguintes itens:
- Python 3.7+
- Biblioteca Ultralytics YOLOv8
- Biblioteca OpenCV
- Conjunto de dados contendo imagens das quatro classes: Araucárias, Capivaras, Gralha Azul e Pinhão.

## Instalação
Primeiramente, é necessário instalar a biblioteca YOLOv8 da Ultralytics. Você pode fazer isso utilizando o pip:

```bash
pip install -r requirements.txt

