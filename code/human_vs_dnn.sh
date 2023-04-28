if [ "$1" == "all" ];
then
    declare -a model_list=("resnet34" "vgg16_bn" "vgg_19_bn")
else
    declare -a model_list=($1)
fi

if [ "$2" == "all" ];
then
    declare -a split_list=(0 1 2 3 4)
else
    declare -a split_list=($2)
fi

for backbone in "${model_list[@]}"
do
    for split in "${split_list[@]}"
    do     
        python trainVal.py -c human_vs_dnn.config --model_id $backbone\_$split --split $split --first_mod $backbone
        retVal=$?
        if [ $retVal -ne 0 ]; then
            exit
        fi
    done
done 