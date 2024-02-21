# ICSE2023Repair
A PyTorch Implementation of ["Tare: Type-Aware Neural Program Repair"](https://xiongyingfei.github.io/papers/ICSE23a.pdf).

# Introduction
Automated program repair (APR) aims to reduce the effort for software development. With the development of deep learning, lots of DL-based APR approaches have been proposed using an encoder-decoder architecture. Despite the promising performance, these models share one same limitation: generating lots of untypable patches. The main reason for this phenomenon is that the existing models do not consider the constraints of code captured by a set of typing rules.

In this paper, we propose, Tare, a type-aware model for neural program repair to learn the typing rules. To encode an individual typing rule, we introduces three novel components: (1) a novel type of grammars, T-Grammar that integrates the type information into a standard grammar, (2) a novel representation of code, T-Graph that integrates the key information needed for type checking in an AST, and (3) a novel type-aware neural program repair approach, Tare that encodes the T-Graph and generates the patches guided by T-Grammar.

The experiment was conducted on three benchmarks, 393 bugs from Defects4J v1.2, 444 additional bugs from Defects4J v2.0, and 40 bugs from QuixBugs. Our results show that Tare repairs 64, 32, and 27 bugs on these benchmarks respectively and outperforms the existing APR approaches on all benchmarks. The further analysis also shows that Tare tends to generate more compilable patches than the existing DL-based APR approaches with the typing rule information.


# The Main File Tree

```
.
├── generation: code for patch generation
│   ├── bugs-QuixBugs
│ 	├── location
│ 	├── location2
│ 	└── tesetDefect4j.py
├── model: code for neural model
│   ├── data
│   ├── Model.py
│   └── relAttention.py
├── train: code for training the model
│   ├── run.py
│   ├── Dataset.py
│   └── SearchNode.py
├── validation: code for patch validation
│   ├── patches-all: plausible patches
│   ├── Dataset.py
│   └── SearchNode.py
```
# Dataset
## Train set
The raw data https://drive.google.com/drive/folders/1ECNX98qj9FMdRT2MXOUY6aQ6-sNT0b_a?usp=sharing from [Recoder](https://github.com/pkuzqh/Recoder).
## Test set
### [Defects4J](https://github.com/rjust/defects4j)
### [QuixBugs](https://github.com/jkoppel/QuixBugs)

# Updated Result
❗❗❗ We test Tare with the perfect fault localization on Defects4J 2.0.

| Project          | Bugs | TBar | SimFix | Recoder | RewardRepair | Tare |**Tare+PerfectLocation**|
|------------------|------|------|--------|---------|--------------|------|--------------------|
||Original Result|Original Result|Original Result|Original Result|Original Result|Original Result|Updated Result|
| Cli              | 39   | 1/7  | 0/4    | 3/3     | 2/-          | **5/13** |**8**|
| Closure          | 43   | 0/5  | **1/5**| 0/7     | **1/-**      | 0/5   |**1**|
| JacksonDatabind  | 112  | 0/0  | 0/0    | 0/0     | 3/-          | 0/4   |0|
| Codec            | 18   | 2/6  | 0/2    | 2/2     | 3/-          | 3/7  |**5**|
| Collections      | 4    | 0/1  | 0/1    | 0/0     | 0/-          | 0/0   |0|
| Compress         | 47   | 1/13 | 0/6    | 3/9     | 0/-          | **4/13** |**4**|
| Csv              | 16   | 0/2  | 1/5    | 4/4     | 2/-          | **5/7**  |**5**|
| JacksonCore      | 26   | 0/6  | 0/0    | 0/4     | 1/-          | 2/7  |**14**|
| Jsoup            | 93   | 3/7  | 1/5    | 7/13    | 4/-          | 10/16|**14**|
| JxPath           | 22   | 0/0  | 0/0    | 0/4     | **3/-**      | 2/10  |**3**|
| Gson             | 18   | 0/0  | 0/0    | 0/0     | **1/-**      | **1/1**   |**1**|
| JacksonXml       | 6    | 0/0  | 0/0    | 0/0     | 0/-          | 0/1   |0|
| **Total**        | 444  | 8/50 | 2/25   | 19/46   | 20/-         | 32/84|**55**|




# Usage

## Train a New Model
```python
CUDA_VISIBLE_DEVICES=0,1 python3 train/run.py train
```
The saved model is ```checkModel/best_model.cpkt```.

## Test the Model
### Generate Patches for Defects4J v1.2 with Ochiai by
```python
CUDA_VISIBLE_DEVICES=0 python3 generation/testDefect4j.py bugid
```

The generated patches are in folder ```generation/patch-all/patch/``` in json.

### Generate Patches for Defects4J v1.2 with groundtruth by
```python
CUDA_VISIBLE_DEVICES=0 python3 generation/testDefect4j1.py bugid
```

The generated patches are in folder ```generation/patch-all/patchground/``` in json.

### Generate Patches for Defects4J v2.0 by
```python
CUDA_VISIBLE_DEVICES=0 python3 generation/testDefects4j2.py bugid
```

The generated patches are in folder ```generation/patch-all/patch2``` in json.

### Generate Patches for QuixBugs by
```python
CUDA_VISIBLE_DEVICES=0 python3 generation/testQuixbug.py bugid
```

The generated patches are in folder ```generation/patch-all/patchQuix``` in json.

### Validate Patches for Defects4J v1.2 with Ochiai
```python
python3 validation/repair.py bugid
```

The results are in folder ```validation/patch-all/patches/``` in json.

### Validate Patches for Defects4J v1.2 with groundtruth
```python
python3 validation/repair1.py bugid
```

The results are in folder ```validation/patch-all/patchesground/``` in json.

### Validate Patches for Defects4J v2.0 with Ochiai
```python
python3 validation/repair2.py bugid
```

The results are in folder ```validation/patch-all/patches2/``` in json.

### Validate Patches for QuixBugs with Ochiai
```python
python3 validation/repairqfix.py bugid
```

The results are in folder ```validation/patch-all/patchesqfix/``` in json.

### Gnerated Patches
The generated patches are in the folder [patches-all](https://github.com/ICSE23Repair/ICSE23Repair/tree/main/validation/patches-all).


# Dependency
* Python 3.7
* PyTorch 1.3
* Defects4J
* Java 8
* docker
* nvidia-docker

