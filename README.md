# CREC

Official implementation of the paper "Revisiting Counterfactual Problems in Referring Expression Comprehension" [CVPR 2024]



## Updates

- [x] (2024/3/6) Release our C-REC datasets C-RefCOCO/+/g.
- [x] (2024/10/18) Release our C-REC model.


## Datasets

**C-RefCOCO/+/g** are three fine-grained counterfactual referring expression comprehension (C-REC) datasets built on three REC benchmark datasets RefCOCO/+/g through our proposed CSG method.

The number of *normal* and *counterfactual* samples in C-RefCOCO/+/g is 1:1. The size of C-RefCOCO/+/g is shown as follows. 

|            | train | val   | testA(test) | testB |
| ---------- | ----- | ----- | ----------- | ----- |
| C-RefCOCO  | 61870 | 15566 | 6994        | 8810  |
| C-RefCOCO+ | 59962 | 15328 | 7846        | 7108  |
| C-RefCOCOg | 30298 | 3676  | 7122        |       |

The number of seven categories of attributes in *normal* samples are shown as follows. Note that there are some splits that do not contain certain categories of attribute words, such as A5 (relative location relation) and A6 (relative location object) in C-RefCOCO+.

|            | A1    | A2   | A3   | A4    | A5   | A6   | A7   |
| ---------- | ----- | ---- | ---- | ----- | ---- | ---- | ---- |
| C-RefCOCO  | 23862 | 5136 | 464  | 16142 | 131  | 131  | 754  |
| C-RefCOCO+ | 28573 | 9864 | 1685 | 2646  | 0    | 0    | 2354 |
| C-RefCOCOg | 11312 | 4114 | 638  | 4024  | 108  | 108  | 244  |



### Dataset Instructions

1. Download [ms-coco train2014 images](https://pjreddie.com/projects/coco-mirror), where the images in our datasets are all from.
2. Our datasets are in:

```
$ROOT/data
|-- crec
    |-- c_refcoco.json
    |-- c_refcoco+.json
    |-- c_refcocog.json
```

3. Definitions of every term in json files: 


| item    | type | description                                                  |
| ------- | ---- | ------------------------------------------------------------ |
| atts    | str  | attribute words                                              |
| bbox    | list | bounding box ([0,0,0,0] for counterfactual samples)          |
| iid     | int  | image id (from ms-coco train2014)                            |
| refs    | str  | the original positive expression for both normal and counterfactual samples |
| cf_id   | int  | counterfactual polarity (**1: counterfactual; 0: normal**)   |
| att_pos | int  | position of attribute words (start from 0)                   |
| query   | str  | text query                                                   |
| neg     | str  | negative query (it would be the normal text for counterfactual samples; this is for contrastive loss calculation) |
| att_id  | int  | category of attribute word, from 1 to 7 (A1-A7)              |


## Model

Our CREC model is based on one-stage referring expression comprehension model, augmented by our newly-built datasets. 

<p align="center">
	<img src="./docs/CREC.png" width="550">
</p>



### Environment Installation
- Python 3.7
- PyTorch 1.11.0 + CUDA 11.3
- Install mmcv following the [installation guide](https://github.com/open-mmlab/mmcv#installation)
- Install [Spacy](https://spacy.io/) and initialize the [GloVe](https://github-releases.githubusercontent.com/84940268/9f4d5680-4fed-11e9-9dd2-988cce16be55?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210815%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210815T072922Z&X-Amz-Expires=300&X-Amz-Signature=1bd1bd4fc52057d8ac9eec7720e3dd333e63c234abead471c2df720fb8f04597&X-Amz-SignedHeaders=host&actor_id=48727989&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_vectors_web_lg-2.1.0.tar.gz&response-content-type=application%2Foctet-stream) and install other requirements as follows:
```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```


### Training and Evaluation

1. **Config preparation**. 
Modify the [config file](./configs/crec_refcoco.py) according to your needs.


2. **Train the model**. 
Download pretrained weights of visual backbone following [Pretrained Weights](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md).

[Optional] Resume from checkpoint:

- To auto resume from `last_checkpoint.pth`, set `train.auto_resume.enabled=True` in config.py, which will automatically resume from `last_checkpoint.pth` saved in `cfg.train.output_dir`.

- To resume from a checkpoint, set `train.auto_resume.enabled=False` and `train.resume_path=path/to/checkpoint.pth` in config.py.

To train our model on `4` GPUs, run:

```shell
bash tools/train.sh configs/crec_refcoco.py 4
```

3. **Test the model.** 
To test our model from `path/to/checkpoint.pth` on `1` GPUs, run:.

```shell
bash tools/eval.sh configs/crec_refcoco.py 1 path/to/checkpoint.pth
```


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Citation

If this repository is helpful for your research, or you want to refer the provided results in your paper, consider cite:
```BibTeX
@inproceedings{yu2024revisiting,
  title={Revisiting Counterfactual Problems in Referring Expression Comprehension},
  author={Yu, Zhihan and Li, Ruifan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13438--13448},
  year={2024}
}
```

## Acknowledgement

Thanks a lot for the nicely organized code from the following repos:
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [SimREC](https://github.com/luogen1996/SimREC)

