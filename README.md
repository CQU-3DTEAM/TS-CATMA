# TS-CATMA

You can find our paper here: [TS-CATMA: A Lung Cancer Electronic Nose Data Classification Method Based on Adversarial Training and Multi-scale Attention](https://link.springer.com/chapter/10.1007/978-981-96-0119-6_7).

## Environment Setup

```
conda create -n tscatma python=3.8.18
cd TS-CATMA
pip install -r ./requirements.txt
```

## Model Training

1. Set the dataset path in `./data_loader.py`.
2. Use the default configuration file or provide your own in `./configs/`
3. Run the following command to start training:
   ```
   python main.py
   ```
   The results will be saved in the `./output/` directory.
4. (Optional) You can set the path to a trained model in each `./plot*.py` script to generate various visualization results.


## Reference

If you use this work in your research, please cite the following paper:
```
@InProceedings{10.1007/978-981-96-0119-6_7,
  author="Chen, Yuze
  and Yi, Lin
  and Wang, Shidan
  and Tian, Fengchun
  and Liu, Ran",
  editor="Hadfi, Rafik
  and Anthony, Patricia
  and Sharma, Alok
  and Ito, Takayuki
  and Bai, Quan",
  title="TS-CATMA: A Lung Cancer Electronic Nose Data Classification Method Based on Adversarial Training and Multi-scale Attention",
  booktitle="PRICAI 2024: Trends in Artificial Intelligence",
  year="2025",
  publisher="Springer Nature Singapore",
  address="Singapore",
  pages="73--78",
  abstract="Accurate lung cancer diagnosis is crucial for effective treatment and improved outcomes. This study introduces TS-CATMA (Time Series Classification with Adversarial Training and Multi-scale Attention), a novel method designed for lung cancer detection using electronic nose data. TS-CATMA leverages a multi-scale attention mechanism and adversarial training to extract discriminative, domain-invariant features from raw time series data. Evaluated on a lung cancer electronic nose dataset, TS-CATMA achieved a detection accuracy of 90.59{\%} with rapid training (6.15 s) and testing (39.57 ms) times, indicating its potential for early diagnosis. The source code is available at https://github.com/CQU-3DTEAM/TS-CATMA.",
  isbn="978-981-96-0119-6"
}
```
