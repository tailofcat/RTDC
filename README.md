# A robust transductive distribution calibration method for semi-supervised few-shot learning

## Environment

- Python==3.7.16
- numpy==1.17.2
- matplotlib==3.1.1
- tqdm==4.36.1
- torchvision==0.6.0
- torch==1.5.0
- Pillow==7.1.2

## Backbone Training

We use the same backbone network and training strategies as 'S2M2_R'. Please refer to https://github.com/nupurkmr9/S2M2_fewshot for the backbone training.


## Extract and save features

After training the backbone as 'S2M2_R', extract features as below:

- Create an empty 'checkpoints' directory.

- Run:
```save_features
python save_plk.py --dataset [miniImagenet/tieredImagenet/CUB/cifar] 
```
### Or you can just use the extracted features/trained models from the folder named "checkpoints_backup"

When using the extracted features, please adjust your file path according to the code.


## Evaluate our Robust Transductive Distribution Calibration (RTDC)

To evaluate our method, run:

```eval
python evaluate_RTDC.py --dataset [miniImagenet/tieredImagenet/CUB/cifar] 
```

## Reference

[wyharveychen/CloserLookFewShot: source code to ICLR'19, 'A Closer Look at Few-shot Classification' (github.com)](https://github.com/wyharveychen/CloserLookFewShot)

[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)

[https://github.com/nupurkmr9/S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)

https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration

