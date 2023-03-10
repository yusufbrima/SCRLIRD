### qsub -v PATH -v PYTHONPATH=$PWD -cwd -l mem=16G,cuda=1,cuda_cores=10000 PretrainConst.py


python3 train.py build --input_dir /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/Downloads/16000_pcm_speeches --output train_dev --n 3


python3 convert.py --input_dir /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/Downloads/wav --output /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/Downloads/VoxCeleb


CUDA_VISIBLE_DEVICES=0 python3 pretrainsize.py --d_input_dir /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/Downloads/16000_pcm_speeches --d_output pcm_speeches --pbs 265 --npt 2 --nds 2 --dbs 64 --epochs 100 --ft 1



qsub -N Trial1 -v CUDA_VISIBLE_DEVICES=0 -v PATH -b y -cwd -l cuda=1,mem=8G,nv_mem_total=8G python3 pretrainsize.py --d_input_dir /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/Downloads/16000_pcm_speeches --d_output pcm_speeches --pbs 265 --npt 2 --nds 2 --dbs 64 --epochs 100 --ft 1


qsub -N voxceleb1 -v CUDA_VISIBLE_DEVICES=0 -v PATH -b y -cwd -l cuda=0,mem=64G python3 train.py build --input_dir  /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/data/learning/VoxCeleb/voxceleb1 --output voxceleb1 --n 2



qsub -N voxceleb1 -v CUDA_VISIBLE_DEVICES=0 -v PATH -b y -cwd -l cuda=0,mem=8G python3 create.py


 CUDA_VISIBLE_DEVICES=0  python3 train.py build --input_dir  /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/data/learning/LibriSpeech100/train  --output train --n 2


  CUDA_VISIBLE_DEVICES=0 python3 train.py train --p_input_dir /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/data/learning/LibriSpeech100/train --p_output  train  --d_input_dir /net/projects/scratch/winter/valid_until_31_July_2023/ybrima/data/learning/PrototypeDS --d_output Prototype --pbs 265 --dbs 64 --epochs 100 --ft 1 --npt 2 --nds 1



