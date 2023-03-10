from pathlib import Path
dataparam = dict(
    HOP_LENGTH =  512,
    FRAME_SIZE =  2048,
    SAMPLE_RATE =  44100,
    NUM_FBANKS = 128, #64
    NUM_FRAMES = 160  # 1 second ~ 100 frames with default params winlen=0.025,winstep=0.01
    )

data = "Prototype"
pretrain = "LibriSpeech100" # Voxceleb1
base_path = "/home/staff/y/ybrima/Desktop/scratch/winter/valid_until_31_July_2023/ybrima/data/learning"
models = "/home/staff/y/ybrima/Desktop/scratch/winter/valid_until_31_July_2023/ybrima/data/Models"
filepaths = dict( 
    pretrain_data_path = Path(f'{base_path}/{pretrain}/train'),
    data_path = Path(f'{base_path}/data/{data}'),
    base_path = Path(f'{base_path}/data/{data}DS/'),
    datapath_path = Path(f'{base_path}/data/{data}DS/{data}'),
    train_file_path = Path(f'{base_path}/data/{data}DS/{data}_Train.npz'),
    val_file_path = Path(f'{base_path}/data/{data}DS/{data}_Val.npz'),
    test_file_path = Path(f'{base_path}/data/{data}DS/{data}_Test.npz'),
    pretrain_file_path = Path(f'{base_path}/{pretrain}/train/Pretrain_{pretrain}.npz'),
    file_path = Path(f'{base_path}/data/{data}DS/{data}_Prototype.npz'),
    result_path = Path(f'./Data/Pretrained_{pretrain}_{data}.csv'),
    model_path = Path(models),
    fig_path = Path('./Figures/'),
    voxceleb  = Path(f"{base_path}/VoxCeleb"),
    librispeech = Path(f"{base_path}/LibriSpeech500/"),
    Public_Speeches = Path(f"{base_path}/Public_Speeches/"),
    pretrain_file_paths =  [Path(f'{base_path}/LibriSpeech100/train'),Path(f'{base_path}/LibriSpeech360/train'),Path(f'{base_path}/LibriSpeech500/train'),Path(f'{base_path}/LibriSpeech960/train'), Path(f"{base_path}/VoxCeleb/VoxCeleb"),Path(f"{base_path}/VoxCeleb/vox2_dev_aac")]
)



learningparams = dict( 
    learning_rate = 0.001,
    batch_size = 265,
    ft_batch_size = 32,
    hidden_units = 512, #512
    projection_units = 128,
    num_epochs = 100,
    dropout_rate = 0.5,
    temperature = 0.05
)
