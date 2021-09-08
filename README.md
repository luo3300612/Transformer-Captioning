# Transformer Captioning

This repository contains the code for Transformer-based image captioning. Based
on [meshed-memory-transformer](https://github.com/aimagelab/meshed-memory-transformer), we further optimize the code for
FASTER training without any accuracy decline.

Specifically, we optimize following aspects:

* vocab: we pre-tokenize the dataset so there are no ' '(space token) in vocab or generated sentences.
* Dataloader: we optimize speed of dataloader and achieve 2x~6x speed-up.
* BeamSearch:
    * Make ops parallel in beam_search.py (e.g. loop gather -> parallel gather)
    * Use cheaper ops (e.g. torch.sort -> torch.topk)
    * Use faster and specialized functions instead of general ones
* Self-critical Training
    * Compute Cider by index instead of raw text
    * Cache tf-idf vector of gts instead of computing it again and again
    * drop on-the-fly tokenization since it is too SLOW.
* contiguous model parameter
* other details...

speed-up result (1 GeForce 1080Ti GPU, num_workers=8, batch_size=50(XE)/100(SCST))

|Training its/s|Original|Optimized|Accelerate|
|---|---|---|---|
|XE|7.5|10.3|138%|
|SCST|0.6|1.3|204%|

|Dataloader its/s|Original XE|Optimized XE|Accelerate|Original SCST|Optimized SCST|Accelerate|
|---|---|---|---|---|---|---|
|batch size=50|12.5|52.5|320%|29.3|90.7|209%|
|batch size=100|5.5|33.5|510%|22.3|88.5|297%| 
|batch size=150|3.7|25.4|580%|13.4|71.8|435%| 
|batch size=200|2.7|20.1|650%|11.4|54.1|376%|

Things I have tried but not useful

* TorchText n-gram counter: slower than the original one.
* nn.Module.MultiHeadAttention: slightly faster than original one.
* GPU cider: very slow
* BeamableMM: slower than the original

## Environment setup

Clone the repository and create the `m2release` conda environment using the `environment.yml` file:

```
conda env create -f environment.yml
conda activate m2release
```

Then download spacy data by executing the following command:

```
python -m spacy download en
```

Note: Python 3.6 is required to run our code.

## Data preparation

To run the code, annotations and detection features for the COCO dataset are needed. Please download the annotations
file [annotations.zip](https://drive.google.com/file/d/1i8mqKFKhqvBr8kEp3DbIh9-9UNAfKGmE/view?usp=sharing) and extract
it.

Detection features are computed with the code provided by [1]. To reproduce our result, please download the COCO
features file [coco_detections.hdf5](https://drive.google.com/open?id=1MV6dSnqViQfyvgyHrmAT_lLpFbkzp3mx) (~53.5 GB), in
which detections of each image are stored under the `<image_id>_features` key. `<image_id>` is the id of each COCO
image, without leading zeros (e.g. the `<image_id>` for `COCO_val2014_000000037209.jpg` is `37209`), and each value
should be a `(N, 2048)` tensor, where `N` is the number of detections.

REMEMBER to do pre-tokenize
```bash
python pre_tokenize.py
```

## Evaluation

Run `python test.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |

## Training procedure

Run `python train.py` using the following arguments:

| Argument | Possible values |
|------|------|
| `--exp_name` | Experiment name|
| `--batch_size` | Batch size (default: 10) |
| `--workers` | Number of workers (default: 0) |
| `--head` | Number of heads (default: 8) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to detection features file |
| `--annotation_folder` | Path to folder with COCO annotations |
| `--logs_folder` | Path folder for tensorboard logs (default: "tensorboard_logs")|

For example, to train our model with the parameters used in our experiments, use

We recommend to use batch size=100 during SCST stage. Since it will accelerate convergence without obvious accuracy decline  
```
python train.py --exp_name test --batch_size 50 --head 8 --features_path ~/datassd/coco_detections.hdf5 --annotation_folder annotation --workers 8 --rl_batch_size 100 --image_field FasterImageDetectionsField --model transformer --seed 118
```

#### References

* [meshed-memory-transformer](https://github.com/aimagelab/meshed-memory-transformer)
* [contiguous](https://github.com/PhilJd/contiguous_pytorch_params)