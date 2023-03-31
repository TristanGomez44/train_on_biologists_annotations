case $1 in
  "noneRed2")
    python trainVal.py -c $2 --model_id noneRed2   --compute_ece True  --compute_masked True
    ;;
 "noneRed_focal2")
    python trainVal.py -c $2 --model_id noneRed_focal2  --focal_weight 1 --nll_weight 0 --loss_on_masked True --sal_metr_mask True  
    ;;
 "noneRed_onlyfocal")
    python trainVal.py -c $2 --model_id noneRed_onlyfocal  --focal_weight 1 --nll_weight 0    --compute_masked True --compute_ece True
    ;;
 "noneRed_onlylossonmasked")
    python trainVal.py -c $2 --model_id noneRed_onlylossonmasked  --loss_on_masked True --sal_metr_mask True  --compute_masked True --compute_ece True
    ;;
  "*")
    echo "no such model"
    ;;
esac

