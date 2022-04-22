A project trying to replicate [SSRN3776915](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3776915).

### Directory Structure

```
.
├── README.md
├── raw # raw intraday price date. Can be downloaded from 链接: https://pan.baidu.com/s/1M1Fksa8Skj5dVI6MpholMA 提取码: 2ybm。
├── src # source code
└── stash # intermediate results stash
```

### Usage

It's recommended to use [Anaconda](https://www.anaconda.com/) to resolve dependencies.

```bash
# install pytorch
conda install pytorch torchvision torchaudio -c pytorch
```

```bash
conda activate
cd src
python3 feature.py
python3 estimate.py
python3 nn.py
```
