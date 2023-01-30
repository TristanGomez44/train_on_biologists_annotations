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
    python trainVal.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask   --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2  --att_metr_mask True --att_metr_mask_start_epoch 0
    ;; 
  "2_noneRed_attMetrMask_test")
    python trainVal.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask_test   --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2  --att_metr_mask True --att_metr_mask_start_epoch 0 --start_mode scratch
    ;; 
  "2_noneRed_attMetrMask_easyStart")
    python trainVal.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask_easyStart --resnet_bilinear False  --big_images True  --stride_lay3 2 --stride_lay4 2 --att_metr_mask True --att_metr_mask_start_epoch 10
    ;;  
  "*")
    echo "no such model"
    ;;
esac