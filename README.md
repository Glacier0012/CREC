# CREC
Official implementation of the paper "Revisiting Counterfactual Problems in Referring Expression Comprehension"



# Updates

- [x] (2024/3/6) Release our C-REC datasets C-RefCOCO/+/g.



# Datasets

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



## Instructions

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

- 'atts': attribute words
- 'bbox': bounding box ([0,0,0,0] for counterfactual samples)
- 'iid': image id (from ms-coco train2014)
- 'refs': the original positive expression for both normal and counterfactual samples
- 'cf_id': counterfactual polarity (**1: counterfactual; 0: normal**)
- 'att_pos': position of attribute words (start from 0)
- 'query': text query for this sample
- 'neg': negative query for this sample (it would be the normal text for counterfactual sample; this is for contrastive loss calculation)
- 'att_id': category of attribute word, from 1 to 7 (A1-A7)


