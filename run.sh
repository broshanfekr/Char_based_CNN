#!/bin/bash  
 python3 train.py -input_dataset_path yelp_academic_dataset_review.txt -output_path myresult.txt -max_seq_len_cutoff 1014 -n_classes 5 -batch_size 100 -num_epochs 20 -evaluate_every 3 -checkpoint_every 2
