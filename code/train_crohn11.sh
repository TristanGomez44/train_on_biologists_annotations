case $1 in
  "brnpa")
    python trainVal.py -c model_crohn11.config --model_id brnpa --resnet_bilinear True --bil_cluster True
    ;; 
  "bcnn")
    python trainVal.py -c model_crohn11.config --model_id bcnn --resnet_bilinear True    
    ;;  
  "noatt")
    python trainVal.py -c model_crohn11.config --model_id noatt    
    ;; 
  "noattNCE")
    python trainVal.py -c model_crohn11.config --model_id noattNCE --sal_metr_mask True --nce_weight 1 --start_mode fine_tune --init_path ../models/CROHN11/modelnoatt_best_epoch43     
    ;;   
  "*")
    echo "no such model"
    ;;
esac