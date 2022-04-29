# -*-coding: utf-8 -*-

# off-target prediction
# demo_dataset.txt with two columns (target sequence, off-target sequence)
# 'target sequence': 63bp wild-type sequence (20bp upstream + 20bp target + 3bp PAM + 20bp downstream);
# 'off-target sequence': 63bp off-target sequence (20bp upstream + 20bp off-target + 3bp PAM + 20bp downstream)
python off_target_predict.py demo_dataset.txt ./results


