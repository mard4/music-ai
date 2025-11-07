# FSD50K Dataset Setup

Scaricare e organizzare i dataset 

- [FSD50K](https://www.kaggle.com/datasets/yousirui1/fsd50k?resource=download-directory) 
- [Kaggle API] Token API scaricato da Kaggle e salvato in: ~/.kaggle/kaggle.json

## Requisiti

- [Kaggle account](https://www.kaggle.com/)
- [Kaggle API](https://github.com/Kaggle/kaggle-api) installata e configurata  
```bash
chmod 600 ~/.kaggle/kaggle.json
mkdir -p data/audio
cd data/audio
kaggle datasets download -d yousirui1/fsd50k
```

MUSDB url: https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems