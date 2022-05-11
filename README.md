# COMM-EXPL with KG-A2C on Text-based Games

Code for ACL2022 paper [Fire Burns, Sword Cuts: Commonsense Inductive Bias for Exploration in Text-based Games]().

## Dependency

- Python 3.7.9
    - pytorch 1.4.0
    - gym 0.17.2
    - jericho 2.4.2
    - networkx 2.4
    - redis 3.4.1
- Redis 4.0.14 
- Standford CoreNLP 3.9.2

## Set-up

- We follow the set-up from [rajammanabrolu/KG-A2C](https://github.com/rajammanabrolu/KG-A2C), and put all the non-python files in ``etc`` directory.

- Run the code
```python
python train.py --do_comm_expl --do_entropy_threshold
```

## Citation

```
@inproceedings{dkryu2022acl,
    title = "Fire Burns, Sword Cuts: Commonsense Inductive Bias for Exploration in Text-based Games",
    author = "Ryu, Dongwon Kelvin and Shareghi, Ehsan and Fang, Meng and Xu, Yunqiu and Pan, Shirui and Haffari, Gholamreza",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```