
#5 4 88 6 6 model_biggpu.config

K=$1
tr_len=$3
val_len=$4
test_len=$5

k=$2

tr_s=$((100*k/K))
tr_e=$(((tr_s+tr_len)%100))

val_s=$tr_e
val_e=$(((val_s+val_len)%100))

test_s=$val_e
test_e=$(((test_s+test_len)%100))

echo $tr_s $val_s $test_s

getBestPath() {
	best_path=$(ls ../models/cross_val2/model$1*best*)
	best_path=($best_path)
	best_path=${best_path[0]}
	echo $best_path
}

#python trainVal.py -c $6 --exp_id cross_val2 --model_id res18_bs16_trlen10_cv$k --train_part_beg $tr_s --val_part_beg $val_s --test_part_beg $test_s \
#																																					 --train_part_end $tr_e --val_part_end $val_e --test_part_end $test_e \
#																																					 --temp_mod linear --feat resnet18

#best_path_res18="$(getBestPath res18_bs16_trlen10_cv$k)"

python trainVal.py -c $6 --exp_id cross_val2 --model_id r3D18_bs16_trlen10_cv$k --train_part_beg $tr_s --val_part_beg $val_s --test_part_beg $test_s \
                                                                           --train_part_end $tr_e --val_part_end $val_e --test_part_end $test_e \
                                                                           --temp_mod linear --feat r2plus1d_18

#python trainVal.py -c $6 --exp_id cross_val2 --model_id res18LSTM_bs16_trlen10_cv$k --train_part_beg $tr_s --val_part_beg $val_s --test_part_beg $test_s \
#                                                                               --train_part_end $tr_e --val_part_end $val_e --test_part_end $test_e \
#                                                                               --temp_mod lstm --feat resnet18

#python trainVal.py -c $6 --exp_id cross_val2 --model_id res18SC_bs16_trlen10_cv$k   --train_part_beg $tr_s --val_part_beg $val_s --test_part_beg $test_s \
#                                                                               --train_part_end $tr_e --val_part_end $val_e --test_part_end $test_e \
#                                                                               --temp_mod score_conv --feat resnet18 --start_mode fine_tune --init_path $best_path_res18 --strict_init False \
#											 																											 	 --score_conv_ker_size 5 --score_conv_attention False

#python trainVal.py -c $6 --exp_id cross_val2 --model_id res18SA_bs16_trlen10_cv$k   --train_part_beg $tr_s --val_part_beg $val_s --test_part_beg $test_s \
#                                                                               --train_part_end $tr_e --val_part_end $val_e --test_part_end $test_e \
#                                                                               --temp_mod score_conv --feat resnet18 --start_mode fine_tune --init_path $best_path_res18 --strict_init False \
#                                                                               --score_conv_ker_size 5 --score_conv_attention True
