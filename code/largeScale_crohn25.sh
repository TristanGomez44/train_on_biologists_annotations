case $1 in
  "noneRed")
    python trainVal.py -c model_crohn25.config --model_id noneRed --epochs 100 --compute_ece True
    ;;
 "noneRed_focal")
    python trainVal.py -c model_crohn25.config --model_id noneRed_focal --epochs 100 --focal_weight 1 --nll_weight 0 --loss_on_masked True --sal_metr_mask True
    ;;
  "noneRed2")
    python trainVal.py -c model_crohn25.config --model_id noneRed2 --epochs 100  --compute_ece True --max_worse_epoch_nb 10 --compute_masked True
    ;;
  "noneRed2_transf")
    python trainVal.py -c model_crohn25.config --model_id noneRed2_transf --epochs 100  --compute_ece True --max_worse_epoch_nb 10 --compute_masked True --first_mod vit_b_16 --big_images False
    ;;
 "noneRed_focal2")
    python trainVal.py -c model_crohn25.config --model_id noneRed_focal2 --epochs 100 --focal_weight 1 --nll_weight 0 --loss_on_masked True --sal_metr_mask True --max_worse_epoch_nb 10 
    ;;
 "noneRed_focal2_allblack")
    python trainVal.py -c model_crohn25.config --model_id noneRed_focal2_allblack --epochs 100 --focal_weight 1 --nll_weight 0 --loss_on_masked True --sal_metr_mask True --max_worse_epoch_nb 10 --sal_metr_bckgr black
    ;;
 "noneRed_focal2_transf")
    python trainVal.py -c model_crohn25.config --model_id noneRed_focal2_transf --epochs 100 --focal_weight 1 --nll_weight 0 --loss_on_masked True --sal_metr_mask True --max_worse_epoch_nb 10 --first_mod vit_b_16 --big_images False
    ;;
 "noneRed_focal2_otherimage")
    python trainVal.py -c model_crohn25.config --model_id noneRed_focal2_otherimage --epochs 100 --focal_weight 1 --nll_weight 0 --loss_on_masked True --sal_metr_mask True --max_worse_epoch_nb 10  --sal_metr_otherimg True
    ;;
 "noneRed_onlyfocal")
    python trainVal.py -c model_crohn25.config --model_id noneRed_onlyfocal --epochs 100 --focal_weight 1 --nll_weight 0 --max_worse_epoch_nb 10   --compute_masked True --compute_ece True
    ;;
 "noneRed_onlylossonmasked")
    python trainVal.py -c model_crohn25.config --model_id noneRed_onlylossonmasked --epochs 100 --loss_on_masked True --sal_metr_mask True --max_worse_epoch_nb 10 --compute_masked True --compute_ece True
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
