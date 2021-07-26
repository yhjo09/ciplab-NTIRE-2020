## Investigating Loss Functions for Extreme Super-Resolution

[NTIRE 2020](https://data.vision.ee.ethz.ch/cvl/ntire20/) Perceptual Extreme Super-Resolution Submission.

Our method ranked first and second in PI and LPIPS measures respectively.

[[Paper]](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Jo_Investigating_Loss_Functions_for_Extreme_Super-Resolution_CVPRW_2020_paper.pdf) 


## Dependency
- Python 3.6
- PyTorch 1.2
- numpy
- pillow
- tqdm


## Test
1. Clone this repo.
```
git clone https://github.com/kingsj0405/ciplab-NTIRE-2020
```

2. Download pre-trained model and place it to `./model.pth`.
- [NTIRE submission version](https://drive.google.com/file/d/10lu7rJ8JmiqGnq9k8N2iLei0aUAdhGcz/view?usp=sharing)
- [Updated version](https://drive.google.com/file/d/1ugIYMCQK-Rw5jyI6CBB3e9ukMCceb7Lm/view?usp=sharing)

3. Place low-resolution input images to `./input`.

4. Run.
```
python test.py
```
If your GPU memory lacks, please try with option `-n 3` or a larger number.

5. Check your results in `./output`.


## Train
1. Clone this repo.
```
git clone https://github.com/kingsj0405/ciplab-NTIRE-2020
```

2. Prepare training png images into `./train`.

3. Prepare validation png images into `./val`.

4. Open `train.py` and modify user parameters in L22.

5. Run.
```
python train.py
```
If your GPU memory lacks, please try with lower batch size or patch size.


## BibTeX
```
@InProceedings{jo2020investigating,
   author = {Jo, Younghyun and Yang, Sejong and Kim, Seon Joo},
   title = {Investigating Loss Functions for Extreme Super-Resolution},
   booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
   month = {June},
   year = {2020}
}
```


## External codes from
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
- [RRDB (ESRGAN)](https://github.com/xinntao/ESRGAN)
