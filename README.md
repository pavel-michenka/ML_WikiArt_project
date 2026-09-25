# WikiArt Style Classification

Projekt klasifikace uměleckých děl podle výtvarného stylu pomocí konvoluční neuronové sítě v čistém PyTorch. Notebook obsahuje vlastní `Dataset`, `DataLoader`, baseline CNN, trénovací smyčku a kontrolu, že model dokáže přeučit malý vzorek dat.

## Požadavky

- Python 3.10 nebo novější
- Jupyter Notebook nebo VS Code s rozšířením Jupyter
- Kaggle účet a API token
- doporučená instalace: virtuální prostředí

## Instalace

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install kaggle torch torchvision pillow matplotlib jupyter
```


## Stažení datasetu z Kaggle

Kaggle: [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart)

Po rozbalení musí existovat cesta:




## Struktura projektu

```text
.
├── assignment.md
├── milestone_1.ipynb
├── README.md
└── WikiArt/
    └── wikiart/versions/1/
        ├── classes.csv
        └── <složky s uměleckými styly>
```
