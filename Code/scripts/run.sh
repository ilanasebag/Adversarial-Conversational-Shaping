pip install -r requirements.txt
pip install -e .


wget -O daily_dialog.zip http://yanran.li/files/ijcnlp_dailydialog.zip
unzip daily_dialog.zip 
rm daily_dialog.zip 

mv ijcnlp_dailydialog data
unzip data/train.zip -d data
unzip data/test.zip -d data
unzip data/validation.zip -d data

python scripts/pretrain_generator.py --dataset-path data/ --lr 1e-3 
python scripts/pretrain_discriminator.py --dataset-path data/ --lr 1e-3 --partial
python scripts/train_policy_gradient_gan.py --dataset-path data/ --lr 1e-3 --regs --discriminator-steps 5 --generator-steps 1 --teacher-forcing
python scripts/evaluate_generator.py --dataset-path data/ --lr 1e-3 --generator-path generator_policy_gradient.pt
