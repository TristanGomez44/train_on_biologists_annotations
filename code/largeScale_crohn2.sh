case $1 in
  "1")
    python trainVal.py -c model_crohn1.config --model_id clus  --stride_lay3 2 --stride_lay4 2 
    ;;
  "1_mast")
    python trainVal.py -c model_crohn1.config --model_id clus_mast  --master_net True --m_model_id clus 
    ;;
  "2")
    python trainVal.py -c model_crohn2.config --model_id clus  --stride_lay3 2 --stride_lay4 2 
    ;;
  "2_mast")
    python trainVal.py -c model_crohn2.config --model_id clus_mast  --master_net True --m_model_id clus 
    ;;
  "3")
    python trainVal.py -c model_crohn3.config --model_id clus  --stride_lay3 2 --stride_lay4 2 
    ;;
  "3_mast")
    python trainVal.py -c model_crohn3.config --model_id clus_mast  --master_net True --m_model_id clus 
    ;;
  "4")
    python trainVal.py -c model_crohn4.config --model_id clus  --stride_lay3 2 --stride_lay4 2 
    ;;
  "4_mast")
    python trainVal.py -c model_crohn4.config --model_id clus_mast  --master_net True --m_model_id clus 
    ;;
  "5")
    python trainVal.py -c model_crohn5.config --model_id clus  --stride_lay3 2 --stride_lay4 2 
    ;;
  "5_mast")
    python trainVal.py -c model_crohn5.config --model_id clus_mast  --master_net True --m_model_id clus 
    ;;
  "1_big")
    python trainVal.py -c model_crohn1.config --model_id clus_big  --stride_lay3 2 --stride_lay4 2 --big_images True
    ;;
  "2_big")
    python trainVal.py -c model_crohn2.config --model_id clus_big  --stride_lay3 2 --stride_lay4 2 --big_images True
    ;;
  "3_big")
    python trainVal.py -c model_crohn3.config --model_id clus_big  --stride_lay3 2 --stride_lay4 2 --big_images True
    ;;
  "4_big")
    python trainVal.py -c model_crohn4.config --model_id clus_big  --stride_lay3 2 --stride_lay4 2 --big_images True
    ;;
  "5_big")
    python trainVal.py -c model_crohn5.config --model_id clus_big  --stride_lay3 2 --stride_lay4 2 --big_images True
    ;;  
  "1_big_mast")
    python trainVal.py -c model_crohn1.config --model_id clus_big_mast  --big_images True  --master_net True --m_model_id clus_big 
    ;;
  "2_big_mast")
    python trainVal.py -c model_crohn2.config --model_id clus_big_mast  --big_images True  --master_net True --m_model_id clus_big 
    ;;
  "3_big_mast")
    python trainVal.py -c model_crohn3.config --model_id clus_big_mast  --big_images True  --master_net True --m_model_id clus_big 
    ;;
  "4_big_mast")
    python trainVal.py -c model_crohn4.config --model_id clus_big_mast  --big_images True  --master_net True --m_model_id clus_big 
    ;;
  "5_big_mast")
    python trainVal.py -c model_crohn5.config --model_id clus_big_mast  --big_images True  --master_net True --m_model_id clus_big 
    ;;
  "2_big_mast_bp")
    python trainVal.py -c model_crohn2.config --model_id clus_big_mast_bp  --big_images True  --master_net True --m_model_id clus_big  --add_patches True
    ;;  
  "1_noneRed")
    python trainVal.py -c model_crohn1.config --model_id noneRed   --resnet_bilinear False  --stride_lay3 2 --stride_lay4 2  --big_images True
    ;;
  "2_noneRed")
    python trainVal.py -c model_crohn2.config --model_id noneRed   --resnet_bilinear False  --stride_lay3 2 --stride_lay4 2  --big_images True
    ;;  
  "3_noneRed")
    python trainVal.py -c model_crohn3.config --model_id noneRed   --resnet_bilinear False  --stride_lay3 2 --stride_lay4 2  --big_images True
    ;; 
  "4_noneRed")
    python trainVal.py -c model_crohn4.config --model_id noneRed   --resnet_bilinear False  --stride_lay3 2 --stride_lay4 2  --big_images True
    ;; 
  "5_noneRed")
    python trainVal.py -c model_crohn5.config --model_id noneRed   --resnet_bilinear False  --stride_lay3 2 --stride_lay4 2  --big_images True
    ;; 
 "1_none_mast")
    python trainVal.py -c model_crohn1.config --model_id none_mast   --resnet_bilinear False    --big_images True --master_net True --m_model_id noneRed   
    ;;
  "2_none_mast")
    python trainVal.py -c model_crohn2.config --model_id none_mast   --resnet_bilinear False    --big_images True --master_net True --m_model_id noneRed 
    ;;  
  "3_none_mast")
    python trainVal.py -c model_crohn3.config --model_id none_mast   --resnet_bilinear False    --big_images True --master_net True --m_model_id noneRed
    ;; 
  "4_none_mast")
    python trainVal.py -c model_crohn4.config --model_id none_mast   --resnet_bilinear False    --big_images True --master_net True --m_model_id noneRed
    ;; 
  "5_none_mast")
    python trainVal.py -c model_crohn5.config --model_id none_mast   --resnet_bilinear False    --big_images True --master_net True --m_model_id noneRed
    ;; 
  "2_noneRed_attMetrMask")
    python trainVal.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask   --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2  --sal_metr_mask True --sal_metr_mask_start_epoch 0
    ;; 
  "2_noneRed_attMetrMask_test")
    python trainVal.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask_test   --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2  --sal_metr_mask True --sal_metr_mask_start_epoch 0 --start_mode scratch
    ;; 
  "2_noneRed_attMetrMask_easyStart")
    python trainVal.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask_easyStart --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --sal_metr_mask_start_epoch 10
    ;;  
  "noneRed_attMetrMask_FT")
    python trainVal.py -c model_crohn2.config --model_id noneRed_attMetrMask_noneRed_attMetrMask_FT --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 100 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46
    ;; 
  "noneRed_0.5attMetrMask_FT")
    python trainVal.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 100 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46 --sal_metr_mask_prob 0.5
    ;; 
  "noneRed_0.5attMetrMask_FT_simclr")
    python trainVal.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT_simclr --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 100 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46 --sal_metr_mask_prob 0.5 --sal_metr_mask_weight 1
    ;;
  "noneRed_attMetrMask_FT_simclr_long")
    python trainVal.py -c model_crohn2.config --model_id noneRed_attMetrMask_FT_simclr_long --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 500 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46 --sal_metr_mask_prob 1 --sal_metr_mask_weight 1
    ;;
  "noneRed_0.5attMetrMask_FT_0.0016simclr")
    python trainVal.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT_0.0016simclr --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 100 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46 --sal_metr_mask_prob 0.5 --sal_metr_mask_weight 0.0016
    ;; 
  "noneRed_attMetrMask_FT_0.0016simclr_test")
    python trainVal.py -c model_crohn2.config --model_id noneRed_attMetrMask_FT_0.0016simclr_test --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 100 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46 --sal_metr_mask_weight 0.0016 --debug True
    ;; 
  "noneRed_0.5attMetrMask_FT_remMaskedObj")
    python trainVal.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT_remMaskedObj --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 100 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46 --sal_metr_mask_prob 0.5  --sal_metr_mask_remove_masked_obj True
    ;;  
  "noneRed_long")
    python trainVal.py -c model_crohn2.config --model_id noneRed_long   --resnet_bilinear False  --stride_lay3 2 --stride_lay4 2  --big_images True --epochs 500 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_best_epoch46
    ;;  
  "noneRed_attMetrMask_FT_0.0016simclr")
    python trainVal.py -c model_crohn2.config --model_id noneRed_attMetrMask_FT_0.0016simclr --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --sal_metr_mask True --epochs 500 --start_mode fine_tune --init_path ../models/CROHN2/modelnoneRed_long_copy_best_epoch111 --sal_metr_mask_weight 0.0016
    ;; 
  "*")
    echo "no such model"
    ;;
esac

