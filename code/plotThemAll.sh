
cd ../results/curves/nopretrai_res9/

python ../../../code/plotAccuracy.py --csv_list run-Accuracy_res18_bs16_trLen10_train-tag-Accuracy.csv         run-Accuracy_res18_bs16_trLen10_val-tag-Accuracy.csv \
                                                run-Accuracy_res18_bs16_trLen10_nopretr_train-tag-Accuracy.csv run-Accuracy_res18_bs16_trLen10_nopretr_val-tag-Accuracy.csv \
                                     --outfile ../../../vis/curves/nopretr.png \
                                     --yaxis Accuracy --xaxis Epochs \
                                     --title "Effect of pretraining on ImageNet" \
                                     --label_list Train Val Train Val --color_list red red blue blue --linestyle_list ":" "-" ":" "-"  \
                                     --leg_nb 2 --leg_names "Pretrained on ImageNet" "No pretraining"


python ../../../code/plotAccuracy.py --csv_list run-Accuracy_res18_bs16_trLen10_nopretr_train-tag-Accuracy.csv run-Accuracy_res18_bs16_trLen10_nopretr_val-tag-Accuracy.csv \
                                                run-Accuracy_res9_bs16_trLen10_nopretr_train-tag-Accuracy.csv  run-Accuracy_res9_bs16_trLen10_nopretr_val-tag-Accuracy.csv \
                                     --outfile ../../../vis/curves/res9.png \
                                     --yaxis Accuracy --xaxis Epochs \
                                     --title "Use of a smaller CNN" \
                                     --label_list Train Val Train Val --color_list blue blue green green --linestyle_list ":" "-" ":" "-" \
                                     --leg_nb 2 --leg_names "ResNet-18" "ResNet-9"  --epochs 20

cd ../smallVsBig/

python ../../../code/plotAccuracy.py --csv_list run-Accuracy_res18_bs16_trLen10_train-tag-Accuracy.csv      run-Accuracy_res18_bs16_trLen10_val-tag-Accuracy.csv \
                                                run-Accuracy_res18_bs16_trLen10_big2_train-tag-Accuracy.csv run-Accuracy_res18_bs16_trLen10_big2_val-tag-Accuracy.csv  \
                                     --outfile ../../../vis/curves/smallVSbig.png \
                                     --yaxis Accuracy --xaxis Epochs \
                                     --title "Use of a bigger training dataset" \
                                     --label_list Train Val Train Val --color_list blue blue orange orange --linestyle_list ":" "-" ":" "-" \
                                     --leg_nb 2 --leg_names "80 videos" "650 videos"  --epochs 20

cd ../attention/

python ../../../code/plotAccuracy.py --csv_list run-Accuracy_res18_bs16_trLen1_train-tag-Accuracy.csv run-Accuracy_res18_bs16_trLen1_val-tag-Accuracy.csv \
                                                run-Accuracy_res18_bs16_trLen1_big_fa_init_bigMaps_bigAtt_norelu_train-tag-Accuracy.csv run-Accuracy_res18_bs16_trLen1_big_fa_init_bigMaps_bigAtt_norelu_val-tag-Accuracy.csv \
                                                run-Accuracy_res18_bs16_trLen1_big_fa_init_bigMaps_full_train-tag-Accuracy.csv run-Accuracy_res18_bs16_trLen1_big_fa_init_bigMaps_full_val-tag-Accuracy.csv                                       \
                                     --outfile ../../../vis/curves/attention.png                                      \
                                     --yaxis Accuracy --xaxis Epochs                                      \
                                     --title "Use of spatial attention"                                      \
                                     --label_list Train Val Train Val Train Val --color_list blue blue red red green green \
                                     --linestyle_list ":" "-" ":" "-" ":" "-"                                      \
                                     --leg_nb 3 --leg_names "Basline" "Simple attention" "Complex attention"  --epochs 20
cd ../../../code/
