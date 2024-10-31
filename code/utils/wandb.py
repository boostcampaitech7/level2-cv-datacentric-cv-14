import wandb

def set_wandb(name, detail, configs) :
    wandb.init(
        name=detail,
        project=name,
        config={
                'image_size': configs['image_size'],
                'input_size': configs['input_size'],
                'batch_size': configs['batch_size'],
                'learning_rate': configs['learning_rate'],
                'epoch': configs['max_epoch']
            }
    )