cd /root/autodl-tmp
unzip CrowdHuman_train01.zip -d Crowdhuman/train
unzip CrowdHuman_train02.zip -d Crowdhuman/train
unzip CrowdHuman_train03.zip -d Crowdhuman/train
unzip CrowdHuman_val.zip -d Crowdhuman/val
mv Crowdhuman/train/Images/* Crowdhuman/train
mv Crowdhuman/val/Images/* Crowdhuman/val
