
split_by () {
    string=$1
    separator=$2

    tmp=${string//"$separator"/$'\2'}
    IFS=$'\2' read -a arr <<< "$tmp"
    for substr in "${arr[@]}" ; do
        echo "$substr"
    done
    echo
}

extractEpoch() {
	filename=$(basename $1)
	epoch="$(split_by $filename epoch)"
  epoch=($epoch)
	echo ${epoch[1]}
}

getBestPath() {
	best_path=$(ls ../models/cross_val2/model$1*best*)
	best_path=($best_path)
	best_path=${best_path[0]}
	echo $best_path
}

modelIdList=()
epochList=()

for initFile in ../models/cross_val2/*.ini
do
  if [[ $initFile != *"p5"* ]];then
        model_id=$(basename $initFile .ini)
        bestPath=$(getBestPath $model_id)
        epoch=$(extractEpoch $bestPath)
        modelIdList+=($model_id)
        epochList+=($epoch)
        echo $model_id $epoch
  fi
done



#python processResults.py --eval_model --exp_id cross_val2 --param_agr feat temp_mod score_conv_attention \
#                         --names ResNet ResNet3D ResNet-LSTM ResNet-SC ResNet-SA \
#                         --keys resnet18,linear,False r2plus1d_18,linear,False resnet18,lstm,False resnet18,score_conv,False resnet18,score_conv,True \
#                         --epochs_to_process 1 1 1 1 1 1 1 1 1 1 --model_ids ${modelIdList[@]}

python processResults.py --eval_model --exp_id cross_val2 --param_agr feat temp_mod score_conv_attention \
                        --names ResNet ResNet-3D ResNet-LSTM ResNet-SC ResNet-SA \
                        --keys resnet18,linear,False r2plus1d_18,linear,False resnet18,lstm,False resnet18,score_conv,False resnet18,score_conv,True \
                        --epochs_to_process ${epochList[@]} --model_ids ${modelIdList[@]}
