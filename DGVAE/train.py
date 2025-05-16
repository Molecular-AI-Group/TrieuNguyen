import os 
import os.path as op 
from utils import * 
from torch_geometric.loader import DataLoader



if __name__ == '__main__' : 
    config = get_config()

    print_config(config)

    config, train_set, max_token, vocab_smi, vocab_graph = get_dataset(config)
    
    train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    model, optim, annealer = get_model(config, device)

    for e in range(config.trained_epoch + 1, config.epoch + 1) : 
        model.train() 
        for i, data in enumerate(tqdm(train_loader, desc=f'Training epoch {e}')) : 
            inp_graph, inp_smi, inp_smi_mask, tgt_smi = decode_data(data, vocab_smi, device=device) 
            output = model(inp_graph, inp_smi, inp_smi_mask)
            loss = loss_fn(output, tgt_smi, annealer[e-1], config)

            loss.backward()
            if config.gradient_clipping : torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            optim.step(), optim.zero_grad()
            

        if e >= config.start_save:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            if e % config.save_every == 0 : checkpoint(config, model, optim, e)

        if e >= config.start_generate : 
            os.makedirs(config.generate_dir, exist_ok=True)
            if e % config.generate_every == 0 : generate_molecule(config, model, e, num_gen=30000, batch=500)