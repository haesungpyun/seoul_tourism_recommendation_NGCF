# seoul_tourism_recommendation_NGCF

### Development environment
    OS: ubuntu
    IDE: vim
    GPU: NVIDIA RTX A6000

### Dependency
    Python        >= 3.7
    tokenizers    >= 0.9.4
    torch         >= 1.10.2
    konlpy        >= 0.6.0
    pandas        >= 1.3.5
    numpy         >= 1.21.5

### Directory
    |-- model
        |-- NGCF.py
        |-- bprloss.py
        |-- demo.py
        |-- experiment.py
        |-- lap_list.pkl
        |-- main.py
        |-- matrix.py
        |-- parsers.py
        |-- saved_model_data
        |   |-- NGCF_implicit_15_512_5e-05_1.0_standard_2_23.pth
        |   |-- item_dict_implicit_15_512_5e-05_1.0_standard_2_22.pkl
        |   |-- lap_list_implicit_15_512_5e-05_1.0_standard_2_22.pkl
        |   |-- num_dict.pkl
        |   `-- user_dict_implicit_15_512_5e-05_1.0_standard_2_22.pkl
        `-- utils.py
