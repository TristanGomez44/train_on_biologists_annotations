if [ "$1" == "all" ];
then
    declare -a metr_list=("Deletion" "Insertion" "IIC_AD" "ADD")
else
    declare -a metr_list=($1)
fi


if [ "$2" == "all" ];
then
    declare -a expl_list=("randommap" "topfeatmap" "randomfeatmap" "am" "cam" "gradcam" "gradcampp" "ablationcam" "scorecam")
else
    declare -a expl_list=($2)
fi

if [ "$3" == "all" ];
then
    declare -a model_list=("noneRed2" "noneRed_focal2")
else
    declare -a model_list=($3)
fi

if [ "$4" == "all" ];
then
    declare -a cum_list=("True" "False")
else
    declare -a cum_list=($4)
fi

if [ "$5" == "all" ];
then
    declare -a data_replace_method_list=("black" "blur" "otherimage")
else
    declare -a data_replace_method_list=($5)
fi

for metric in "${metr_list[@]}"
do
    echo $metric
    for expl in "${expl_list[@]}"
    do
        echo ' '$expl
        for model in "${model_list[@]}"
        do     
            echo '  '$model

            for cum in "${cum_list[@]}"
            do  
                echo '   '$cum 
                for method in  "${data_replace_method_list[@]}"
                do       
                    echo '    '$method
                    python compute_scores_for_saliency_metrics.py -c model_crohn25.config --attention_metric $metric --model_id $model --att_metrics_post_hoc $expl --cumulative $cum --data_replace_method $method

                    retVal=$?
                    if [ $retVal -ne 0 ]; then
                        exit
                    fi
                done
            done
        done
    done
done 

python3 compute_saliency_metrics.py -c model_crohn25.config