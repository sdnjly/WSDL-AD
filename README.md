# WSL-Arrhythmia-Detection

This repository contains code for the paper:

**Weakly supervised deep learning for generalizable beat-by-beat arrhythmia detections**

Yang Liu, Qince Li, Kuanquan Wang, Runnan He, Yongfeng Yuan, Jun Liu, Yong Xia, Henggui Zhang

School of Computer Science and Technology, Harbin Institute of Technology (HIT), Harbin, Heilongjiang, China.

**Abstract:**
Supervised learning (SL) methods for beat-by-beat arrhythmia detection suffer from poor generalization ability due to the lack of large-sample and finely-annotated (every beat is labeled) electrocardiogram (ECG) data for model training. Although ECG data with coarse-grained annotations (an ECG recording is labeled as a unit) are readily available, the SL-based models trained on them generally can only classify whole recordings rather than individual heartbeats, which fails to meet some clinical needs, such as measuring the frequency and burden of arrhythmias. In this work, we propose a weakly supervised deep learning framework for arrhythmia detection (WSDL-AD), which permits training a fine-grained (beat-by-beat) arrhythmia detector with large amounts of coarsely annotated ECG data to improve the generalization ability. In this framework, heartbeat classification and recording classification are integrated into a deep neural network for end-to-end training with only recording labels. Several techniques, including knowledge-based features, masked aggregation, and supervised pre-training, are proposed to improve the accuracy and stability of the heartbeat classification under weak supervision. We evaluate our framework for the detection of ventricular ectopic beats and supraventricular ectopic beats on three independent benchmarks according to the recommendations from the Association for the Advancement of Medical Instrumentation (AAMI). Our WSDL-AD model significantly outperforms its SL-based counterpart and also other state-of-the-art methods on these benchmarks. It demonstrates that the WSDL-AD framework can leverage the abundant coarsely-labeled data to achieve a better generalization ability than previous methods while maintaining fine detection granularity. Therefore, this framework has a great potential to be used in clinical and telehealth applications.

![avatar](images/graph_abstract.png)

## Requirements
* Python 3
* Tensorflow (version >= 2.3)
* PyWavelets (version >= 1.1)
* Scikit-learn (version >= 0.24)
* WFDB (version >= 3.2)

## Datasets
* MIT-BIH Arrhythmia Database

  Download the [dataset](https://physionet.org/content/mitdb/1.0.0/).
  
* Physionet/CinC challenge 2021 dataset

  Download the [dataset](https://physionetchallenges.org/2021/)

## Detect the QRSs and store them in files
This script just needs to be run once before the first time of model training.
```
python QRS_detection.py --cinc_path $path_of_cinc_db
```

## Supervised learning (SL) setting
 ```
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
               --supervisedMode 'SL' \
               --denoise --normalization \
               --feature_fusion_with_rrs \
               --feature_fusion_with_entropy \
               --aggreg_type MGMP \
               --log_file results/Fully_supervised.csv \
               --training_number 1
```

## Weakly supervised learning (WSL) setting
```
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
                --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --feature_fusion_with_entropy \
                --aggreg_type MGMP \
                --log_file results/main_WSL.csv \
                --training_number 1
```

## SL + WSL setting
* Recommended model training
```
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --feature_fusion_with_entropy \
                --aggreg_type MGMP --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_ablation_features.csv \
                --training_number 1
``` 
## Ablation studies
* Ablation studies for knowledge-based features
```
# No hand-crafted feature 
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --aggreg_type MGMP --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_ablation_features.csv \
                --training_number 1
```
```
# with only relative RR interval
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --aggreg_type MGMP --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_ablation_features.csv \
                --training_number 1
```
```
# with only RR entropy
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_entropy \
                --aggreg_type MGMP --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_ablation_features.csv \
                --training_number 1
```

* Ablation studies for aggregation
```
# GMP
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --feature_fusion_with_entropy \
                --aggreg_type GMP --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_ablation_aggregation.csv \
                --training_number 1 
```
```
# GAP
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --feature_fusion_with_entropy \
                --aggreg_type GAP --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_ablation_aggregation.csv \
                --training_number 1 
```
```
# LSE
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --feature_fusion_with_entropy \
                --aggreg_type LSE --LSE_r 5 \
                --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_ablation_aggregation.csv \
                --training_number 1 
```
## Assess the training stability in multiple sessions of training 
* Training stability of the WSL setting
```
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --feature_fusion_with_entropy \
                --aggreg_type MGMP \
                --log_file results/main_stability_WSL.csv \
                --training_number 50
``` 
* Training stability of the SL+WSL setting
```
python main.py --cinc_path $path_of_the_training_data \
               --mitdb_path $path_of_mitdb \
               --testset_base_path $home_path_of_testsets \
                --denoise --normalization \
                --feature_fusion_with_rrs \
                --feature_fusion_with_entropy \
                --aggreg_type MGMP --initialize_with_SL \
                --initialize_data_proportion 0.5 \
                --log_file results/main_stability_SL_WSL.csv \
                --training_number 1
``` 