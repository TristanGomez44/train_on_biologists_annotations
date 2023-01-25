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
  "*")
    echo "no such model"
    ;;
esac