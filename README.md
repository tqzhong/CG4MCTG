# CG4MCTG
This is the official implementation for the paper [Benchmarking and Improving Compositional Generalization of Multi-aspect Controllable Text Generation](https://arxiv.org/pdf/2404.04232.pdf) which has been accepted to appear at the main conference of ACL 2024. If you have any questions, please feel free to create an issue or contact the email: ztq602656097@mail.ustc.edu.cn, lizhaoyi777@mail.ustc.edu.cn.

## Info
- About the dataset in compmctg benchmark, please check [data](https://github.com/tqzhong/CG4MCTG/tree/main/data).
- About the meta-mctg framework, please check [meta-mctg](https://github.com/tqzhong/CG4MCTG/tree/main/meta-mctg).
- About the evaluation system, please check [evaluation](https://github.com/tqzhong/CG4MCTG/tree/main/evaluation).
- About the construction of protocols in compmctg benchmark, please check [compmctg_protocols](https://github.com/Zhaoyi-Li21/compmctg_protocols).


## Citation
```
@inproceedings{zhong-etal-2024-benchmarking,
    title = "Benchmarking and Improving Compositional Generalization of Multi-aspect Controllable Text Generation",
    author = "Zhong, Tianqi  and
      Li, Zhaoyi  and
      Wang, Quan  and
      Song, Linqi  and
      Wei, Ying  and
      Lian, Defu  and
      Mao, Zhendong",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.351",
    pages = "6486--6517",
    abstract = "Compositional generalization, representing the model{'}s ability to generate text with new attribute combinations obtained by recombining single attributes from the training data, is a crucial property for multi-aspect controllable text generation (MCTG) methods. Nonetheless, a comprehensive compositional generalization evaluation benchmark of MCTG is still lacking. We propose CompMCTG, a benchmark encompassing diverse multi-aspect labeled datasets and a crafted three-dimensional evaluation protocol, to holistically evaluate the compositional generalization of MCTG approaches. We observe that existing MCTG works generally confront a noticeable performance drop in compositional testing. To mitigate this issue, we introduce Meta-MCTG, a training framework incorporating meta-learning, where we enable models to learn how to generalize by simulating compositional generalization scenarios in the training phase. We demonstrate the effectiveness of Meta-MCTG through achieving obvious improvement (by at most 3.64{\%}) for compositional testing performance in 94.4{\%}.",
}
```
