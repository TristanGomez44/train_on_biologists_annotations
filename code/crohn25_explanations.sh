python compute_explanations.py -c model_crohn25.config --model_id noneRed_focal2    --exp_id CROHN25   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc gradcam --img_nb_per_class 3 --viz_id allposthoc

python compute_explanations.py -c model_crohn25.config --model_id noneRed_focal2    --exp_id CROHN25   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc gradcampp  --img_nb_per_class 3 --viz_id allposthoc

python compute_explanations.py -c model_crohn25.config --model_id noneRed_focal2    --exp_id CROHN25   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc scorecam  --img_nb_per_class 3 --viz_id allposthoc

python compute_explanations.py -c model_crohn25.config --model_id noneRed_focal2    --exp_id CROHN25   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc ablationcam  --img_nb_per_class 3 --viz_id allposthoc