#python3 separability_study.py -c model_emb10.config --model_id noneRed --stride_lay3 2 --stride_lay4 2

#python3 separability_study.py -c model_emb10.config --model_id noneRed --stride_lay3 2 --stride_lay4 2 --transf2 black_patches

#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_blur

#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_size20
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_size60
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_blur_size20
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_blur_size60


#Conclusions: 
#Les bordures gauches et droits des patchs sont détectés commes des zones saillantes par le modèles
#Flouter supprimer ce problème
#en conséquent, la séparabilité se réduit, surtout pour les petits patchs noirs 
#Par contre, on voit que les zones masqués donne une activité plus faible que l'activité minimale.
#Cela réduit la norme du vecteur de feature finale. Est-ce que le SVM utilise ça pour distinguer les deux groupes ?

#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_size20  --normalize 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_size60 --normalize 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_blur_size20 --normalize 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_blur_size60 --normalize 

#Normaliser semble avoir un impact très réduit donc non 

#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 img_size20 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 img_size60 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 img_blur_size20 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 img_blur_size60 

#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_size20_nb90 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_size60_nb10 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 img_blur_size20_nb90 
#python3 separability_study.py -c model_emb10.config --model_id none_mast --stride_lay3 2 --stride_lay4 2 --transf2 img_blur_size60_nb10 

#Img bckgr aide ! 

python3 separability_study.py -c model_emb10.config --model_id none_mast --model_id2 none_mast_spars --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_nb10
python3 separability_study.py -c model_emb10.config --model_id none_mast --model_id2 none_mast_spars --stride_lay3 2 --stride_lay4 2 --transf2 black_patches_nb120
