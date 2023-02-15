if [ "$1" == "all" ];
then
    declare -a metr_list=("Deletion" "Insertion" "IIC_AD" "ADD")
else
    declare -a metr_list=($1)
fi

if [ "$2" == "all" ];
then
    declare -a expl_list=("gradcam" "gradcampp" "ablationcam" "scorecam")
else
    declare -a expl_list=($2)
fi

for metric in "${metr_list[@]}"
do
    echo ---------- $metric -------------
    for expl in "${expl_list[@]}"
    do
        echo $expl
        python compute_scores_for_saliency_metrics.py -c model_crohn11.config --attention_metric $metric --model_id noatt --att_metrics_post_hoc $expl 

        retVal=$?
        if [ $retVal -ne 0 ]; then
            exit
        fi

    done
    python compute_scores_for_saliency_metrics.py -c model_crohn11.config --attention_metric $metric --model_id noatt 
    python compute_scores_for_saliency_metrics.py -c model_crohn11.config --attention_metric $metric --model_id brnpa  --resnet_bilinear True --bil_cluster True
    python compute_scores_for_saliency_metrics.py -c model_crohn11.config --attention_metric $metric --model_id bcnn --resnet_bilinear True   
done 

python3 compute_saliency_metrics.py -c model_crohn11.config