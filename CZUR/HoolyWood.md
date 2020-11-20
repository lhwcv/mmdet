## HollyWood Head

- SSD 300 <br/>
  modify  config's data root in configs/__base__/datasets/hooly_wood_head.py 
  train with 2 GPU: 
  ```
  CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh \
   ./configs/ssd/ssd300_holly_head.py 2 --work-dir /home/lhw/m2_disk/work_dir/holly_ssd300/
  ```
  
  val
  ```
  python tools/test.py \
    ./configs/ssd/ssd300_holly_head.py \
    /home/lhw/m2_disk/work_dir/holly_ssd300/epoch_1.pth \
    --eval mAP
  ```
 
 - CZUR Coarse Person Head Model
  train with 2 GPU: <br/>
  (you may modify the data root in config file)
  ```
  CUDA_VISIBLE_DEVICES=0,1 PORT=29501 ./tools/dist_train.sh \
   ./configs/cz_head/cz_head_coarse.py 2 --work-dir /home/lhw/m2_disk/work_dir/cz_head_coarse_224/
  ```
