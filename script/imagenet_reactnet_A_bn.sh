#Binarize activation
DATADIR=data/imagenet
SAVEDIR1=ReActNet_Qa_bn
SAVEDIR2=ReActNet_Qaw_bn
BS=256
Epoch=256
python -u train.py \
    --data ${DATADIR} \
    --save ${SAVEDIR1} \
    --dataset imagenet \
    --batch_size ${BS} \
    --arch reactnet-A \
    --bn_type bn \
    --loss_type kd \
    --teacher dm_nfnet_f0 \
    --learning_rate 5e-4 \
    --epochs ${Epoch} \
    --weight_decay 1e-5 \
    --agc \
    --clip_value 0.02


#Binarize activation and weight
python -u train.py \
    --data ${DATADIR} \
    --save ${SAVEDIR2} \
    --dataset imagenet \
    --batch_size ${BS} \
    --arch reactnet-A \
    --bn_type bn \
    --binary_w \
    --pretrained ${SAVEDIR1}/model_best.pth.tar \
    --loss_type kd \
    --teacher dm_nfnet_f0 \
    --learning_rate 5e-4 \
    --epochs ${Epoch} \
    --weight_decay 0 \
    --agc \
    --clip_value 0.02








