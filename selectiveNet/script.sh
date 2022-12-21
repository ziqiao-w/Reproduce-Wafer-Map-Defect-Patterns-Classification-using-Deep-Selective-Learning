python main.py --is_train True --data_dir ./train.pkl --train_dir ./train_aug1 >log_aug1
python main.py --is_train True --coverage 0.50 --data_dir ./train.pkl --train_dir ./train_aug1_cov1 >log_aug2
python main.py --is_train True --coverage 0.25 --data_dir ./train.pkl --train_dir ./train_aug1_cov2 >log_aug3
python main.py --is_train True --data_dir ./train.pkl --batch_size 256 --train_dir ./train_aug2 >log_aug4 
python main.py --is_train True --train_dir ./train1 --batch_size 256 >log1
python main.py --is_train True --train_dir ./train2 >log2
