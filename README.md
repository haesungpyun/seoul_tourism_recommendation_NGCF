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
    .
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

### Quick Start
    python demo.py

    .
    |-- Create_userId.py
    |-- GMF.py
    |-- MLP.py
    |-- NeuMF.py
    |-- README.md
    |-- create_congestion.py
    |-- csv_to_pickle.py
    |-- data_utils.py
    |-- dataset
    |   |-- congestion_1_2.csv
    |   |-- congestion_1_2.pkl
    |   |-- destination_id_name_genre_coordinate.csv
    |   |-- destination_id_name_genre_coordinate.pkl
    |   |-- seoul_gu_dong_coordinate.csv
    |   `-- seoul_gu_dong_coordinate.pkl
    |-- demo.py
    |-- evaluate.py
    |-- main.py
    |-- model_congestion
    |   |-- GMF.py
    |   `-- MF.py
    |-- parser.py
    `-- pretrain_model

