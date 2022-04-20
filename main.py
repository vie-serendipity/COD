import os
import pprint
import random
import warnings
import torch
import numpy as np
from trainer import Trainer, Tester

from option import getOption
warnings.filterwarnings('ignore')
opt = getOption()

torch.cuda.set_device(1)

def main(opt):
    print('<---- Training Params ---->')
    pprint.pprint(opt)

    # Random Seed
    seed = opt.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if opt.action =='train':
        save_path = os.path.join(opt.model_path, opt.dataset, f'EG_{str(opt.exp_num)}')

        # Create model directory
        os.makedirs(save_path, exist_ok=True)
        Trainer(opt, save_path)

    else:
        save_path = os.path.join(opt.model_path, opt.dataset, f'EG_{str(opt.exp_num)}')

        datasets = ['COD10K-v3']
        for dataset in datasets:
            opt.dataset = dataset
            test_loss, test_mae, test_maxf, test_avgf, test_s_m = Tester(opt, save_path).test()

            print(f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.4f} '
                  f'| AVG_F:{test_avgf:.4f} | MAE:{test_mae:.4f} | S_Measure:{test_s_m:.4f}')


if __name__ == '__main__':
    main(opt)