from hparams import Hparams
import time
import wandb

def initiate_run(hparams:Hparams):
    """
    Initialize connection to wandb and begin the run using provided hparams
    """
    with open(hparams.keyring_dir + 'wandb.key') as key:
        wandb.login(key=key.read())
        key.close()

    run = wandb.init(
        name=f"{hparams.architecure}_{int(time.time())}",
        project=hparams.project,
        config=hparams.wandb_export()
    )

    