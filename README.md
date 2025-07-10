# HyperHawkes
HyperHawkes: Hypergraph-based Temporal Modelling of Repeated Intent for Sequential Recommendation

Preprocessed datasets already in `datasets` folder. 

## Usage
Ta-feng dataset as example, run:
    ```
    python run.py --dataset ta-feng
    ```

## Requirements
- Python >= 3.8
- PyTorch >= 2.0.1
- torch-geometric >= 2.3.1
- numpy == 1.24.4
- recbole >= 1.2.1
- torch-kmeans >= 0.2.0
- mlxtend >= 0.23.4

## Citation
```
@inproceedings{peintner2025hyperhawkes,
  author       = {Andreas Peintner and
                  Amir Reza Mohammadi and
                  Michael M{\"{u}}ller and
                  Eva Zangerle},
  title        = {Hypergraph-based Temporal Modelling of Repeated Intent for Sequential
                  Recommendation},
  booktitle    = {Proceedings of the {ACM} on Web Conference 2025, {WWW} 2025, Sydney,
                  NSW, Australia, 28 April 2025- 2 May 2025},
  pages        = {3809--3818},
  publisher    = {{ACM}},
  year         = {2025},
  url          = {https://doi.org/10.1145/3696410.3714896},
  doi          = {10.1145/3696410.3714896}
}
```
