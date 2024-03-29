This is a fork of [FarecastTKGQA](https://link.springer.com/chapter/10.1007/978-3-031-47240-4_29 "Link to the original paper").

We have made the code runnable on Windows with `Python 3.8`.

Environment setup:

```markdown
Package            Version
------------------ ------------
certifi            2024.2.2
charset-normalizer 3.3.2
click              8.1.7
colorama           0.4.6
filelock           3.13.3
fsspec             2024.3.1
idna               3.6
Jinja2             3.1.3
joblib             1.3.2
MarkupSafe         2.1.5
mpmath             1.3.0
networkx           3.1
numpy              1.19.0
packaging          24.0
pillow             10.2.0
pip                23.3.1
regex              2023.12.25
requests           2.31.0
sacremoses         0.1.1
scikit-learn       1.3.2
scipy              1.9.3
setuptools         68.2.2
sympy              1.12
threadpoolctl      3.4.0
tokenizers         0.9.4
torch              2.2.2+cu121
torchaudio         2.2.2+cu121
torchvision        0.17.2+cu121
tqdm               4.66.2
transformers       4.2.2
typing_extensions  4.10.0
urllib3            2.2.1
wheel              0.41.2
```

To install the `ForecastTKGQA` environment, simply run `pip install -r requirements.txt`. It is recommended not to use `conda install` because some packages may not be found on `anaconda`. Please note that this environment differs from `TANGO`. Although the official `TANGO` repository requires `torch 1.4`, `torch 2.2.2+cu121` is sufficient to run `TANGO`, so there's **no** need to install `torch 1.4`. For other steps, refer to the readme file in the [`ForecastTKGQA` repository](https://github.com/ZifengDing/ForecastTKGQA) and follow the instructions.
