## Investigating Loss Functions for Extreme Super-Resolution

NTIRE2020 Real World Super-Resolution Challenge Track 1 Submission

[Factsheet](./fact_sheet.pdf)
[Results](https://)


## Test
#### Dependency
- Python 3
- PyTorch 1.2
- numpy
- pillow
- tqdm

#### Test 
1. Clone this repo.
```
git clone 
```

2. Download pre-trained [model](https://drive.google.com/file/d/10lu7rJ8JmiqGnq9k8N2iLei0aUAdhGcz/view?usp=sharing) and place it to `./model.pth`.

3. Place low-resolution input images to `./input`.

4. Run.
```
python test.py
```
If your GPU memory lacks, please try with option `-n 3` or with larger number.

5. Check your results in `./output`.


