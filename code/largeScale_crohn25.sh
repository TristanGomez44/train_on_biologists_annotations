case $1 in
  "noneRed")
    python trainVal.py -c model_crohn25.config --model_id noneRed --epochs 100 --compute_ece True
    ;;
 "noneRed_focal")
    python trainVal.py -c model_crohn25.config --model_id noneRed_focal --epochs 100 --focal_weight 1 --nll_weight 0 --focal_loss_on_masked True --sal_metr_mask True
    ;;
  "noneRed_salMask_FT_linschedsimclr")
    python trainVal.py -c model_crohn25.config --model_id noneRed_salMask_FT_linschedsimclr --sal_metr_mask True --nce_weight_sched True --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epochX
    ;; 
  "noneRed_salMask_FT_linschedsimclr_proj")
    python trainVal.py -c model_crohn25.config --model_id noneRed_salMask_FT_simclr_proj --sal_metr_mask True  --nce_weight_sched True --nce_proj_layer True --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epochX
    ;;
  "noneRed_salMask_FT_adv")
    python trainVal.py -c model_crohn25.config --model_id noneRed_salMask_FT_adv --sal_metr_mask True --adv_weight 1 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epochX 
    ;;
  "*")
    echo "no such model"
    ;;
esac

