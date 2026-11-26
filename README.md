# Álgebra Linear no Processamento de Imagens (Manim)

Apresentação visual modular construída com **Manim Community Edition** ilustrando como conceitos de Álgebra Linear fundamentam o processamento digital de imagens.

## Estrutura do Projeto

```text
├── requirements.txt
└── src/
    ├── config/
    │   └── theme.py                        # Paleta de cores e tipografia
    ├── components/
    │   ├── slide_layout.py                 # Cartões, cabeçalhos e transições
    │   └── pixel_grid.py                   # Grades de pixels (cinza, RGB, canais) e matrizes
    ├── topics/
    │   ├── t01_vectors_matrices_tensors.py # 1. Vetores, Matrizes e Tensores
    │   ├── t02_linear_transformations.py   # 2. Transformações Lineares
    │   ├── t03_eigen_symmetric_corners.py  # 3. Autovalores, Autovetores e Matrizes Simétricas (Harris)
    │   ├── t04_thresholding.py             # 4. Thresholding (Limiarização)
    │   └── t05_convolution_dot_product.py  # 5. Convolução e Produto Escalar (Blur, Nitidez, Contornos)
    └── main.py                             # Orquestrador da apresentação completa e cenas individuais
```

## Ambiente Virtual

Ative a `venv` (já configurada na raiz do repositório) ou execute os binários diretamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Comandos de Renderização

Renderizar a apresentação completa em baixa qualidade (rápido para revisão, `-ql`) ou alta qualidade (`-qh` para 1080p60):

```powershell
.\.venv\Scripts\manim.exe -ql src/main.py FullPresentationScene
.\.venv\Scripts\manim.exe -qh src/main.py FullPresentationScene
```

Renderizar tópicos individuais:

```powershell
.\.venv\Scripts\manim.exe -ql src/main.py IntroScene
.\.venv\Scripts\manim.exe -ql src/main.py VectorsMatricesTensorsScene
.\.venv\Scripts\manim.exe -ql src/main.py LinearTransformationsScene
.\.venv\Scripts\manim.exe -ql src/main.py EigenSymmetricCornersScene
.\.venv\Scripts\manim.exe -ql src/main.py ThresholdingScene
.\.venv\Scripts\manim.exe -ql src/main.py ConvolutionDotProductScene
.\.venv\Scripts\manim.exe -ql src/main.py ConclusionScene
```
