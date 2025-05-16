import datetime
import os.path as op
from utils import * 
from argparse import Namespace

best_model = {
    'TGVAEv2_8H_1': 35,
    'TGVAEv2_8H_2': 32,
    'TGVAEv2_8H_3': 38,
    'TGVAEv2_8H_PE_1': 37,
    'TGVAEv2_8H_PE_2': 34,
    'TGVAEv2_8H_PE_3': 39,
    'TGVAEv2_8H_PE_E100_1': 96,
    'TGVAEv2_8H_PE_E100_2': 99,
    'TGVAEv2_8H_PE_E100_3': 98,
    'DGVAE_4H_1': 37,
    'DGVAE_4H_2': 38,
    'DGVAE_4H_3': 37,
    'DGVAE_4H_PE_1': 35,
    'DGVAE_4H_PE_2': 40,
    'DGVAE_4H_PE_3': 39,
    'DGVAE_4H_PE_E100_1': 96,
    'DGVAE_4H_PE_E100_2': 100,
    'DGVAE_4H_PE_E100_3': 81,
}


if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='TGVAEv2_8H_PE_E100_1')
    parser.add_argument('-e', '--epoch', type=int, default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('--num', type=int, default=30000)
    parser.add_argument('--batch', type=int, default=500)
    arg = parser.parse_args()

    if arg.output : assert (arg.output).endswith('.txt'), 'Output file must be a .txt file'
    assert torch.cuda.is_available(), 'CUDA is not available. Please use a GPU to generate molecules.'

    date = datetime.datetime.now().strftime('%m_%d_%Y')
    hour = datetime.datetime.now().strftime('%H-%M-%S')
    
    os.makedirs(op.join(op.dirname(op.abspath(__file__)), 'data', 'inference', date), exist_ok=True)

    output = f'{arg.model}_E{arg.epoch if arg.epoch else best_model[arg.model]}_{hour}.txt' if arg.output is None else arg.output


    model_folder = op.join(op.dirname(op.abspath(__file__)), 'checkpoint', arg.model)
    config = Namespace(**load(op.join(model_folder, 'config.json')))
    snapshot = load(op.join(model_folder, f'snapshot_{arg.epoch if arg.epoch else best_model[arg.model]}.pt'))


    model = get_model(config, 'cuda', snapshot)
    for _ in tqdm(range(arg.num // arg.batch), desc='Generating') : 
        smi_token = model.generate(config, num_gen=arg.batch)
        smi = convert_token(smi_token, config.vocab_smi)
        save(smi, 
             path=op.join(op.dirname(op.abspath(__file__)), 'data', 'inference', date, output),
             mode='a')

