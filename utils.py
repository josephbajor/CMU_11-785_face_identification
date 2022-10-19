from hparams import Hparams
import time
import wandb

def initiate_run(hparams:Hparams):
    """
    Initialize connection to wandb and begin the run using provided hparams
    """
    with open(hparams.keyring_dir + 'wandb.key') as key:
        wandb.login(key=key.read().strip())
        key.close()

    if hparams.use_wandb:
        mode = 'online'
    else:
        mode = 'disabled'

    run = wandb.init(
        name=f"{hparams.architecture}_{int(time.time())}",
        project=hparams.project,
        config=hparams.wandb_export(),
        mode=mode
    )

    return run

    