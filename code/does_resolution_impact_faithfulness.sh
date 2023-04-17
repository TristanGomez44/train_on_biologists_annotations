
case $1 in
  "noneRed2_lr")
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed2_lr --att_metrics_post_hoc cam --big_images False 
    ;;
  "noneRed_focal2_lr")
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed_focal2_lr --att_metrics_post_hoc cam --big_images False 
    ;;
  "noneRed2")
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed2 --att_metrics_post_hoc cam 
    ;;
  "noneRed_focal2")
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed_focal2 --att_metrics_post_hoc cam 
    ;;
  "all")
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed2_lr --att_metrics_post_hoc cam --big_images False 
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed_focal2_lr --att_metrics_post_hoc cam --big_images False
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed2 --att_metrics_post_hoc cam 
    python does_resolution_impact_faithfulness.py -c model_crohn25.config --model_id noneRed_focal2 --att_metrics_post_hoc cam 
    ;;
  "*")
    echo "no such model"
    ;;
esac

