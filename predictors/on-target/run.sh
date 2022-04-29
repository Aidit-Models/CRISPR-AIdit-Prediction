# -*-coding: utf-8 -*-


# on-target prediction of target sequence
# for prediction of a single target sequence
python on_target_predict.py K562 ATTCACGAAGGGCTGCAGGAAGCGTACCCCCAGGTCTTGCAGGTCCTCGGGAGGCTTCACCTC


# for batch prediction of target sequences
# demo_datasets.txt with column target sequence (63bp: 20bp upstream + 20bp target + 3bp PAM + 20bp downstream)
python on_target_predict.py K562 demo_dataset.txt ./results


