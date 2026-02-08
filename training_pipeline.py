from imports import * 

@dataclass
class TRAINING_CONFIG:
    '''
    This is the TRAINING_CONFIG dataclass
    it will be used to define the following
    
    dataset
    classification model
    LLM-translator
    main_config which contains
        num epochs, 
        batch size, 
        lr scheduler config, 
        num workers and all
    wandb config has the name of the wandb init
        like project name, 
        run name etc.
    '''
    seed: int = 42
    run: int = 1
    device: str = 'cuda'
    MAIN: Dict = {
        'scheduler_class': CosineAnnealingLR,
        'num_epochs': 200,
        'warmup_epochs': 10,
        'optimizer_class': SGD(
            
        ),
        'optimizer_config':{
            'lr':1e-2,
            'momentum':0.9,
            'nesterov':True,
            'dampening':False,
            'weight_decay':0.0005
        }
    }
    DATASET: Dict = {
        'batch size': 512,
        'num_workers': 2,
    }
    WANDB: Dict = {
        'project-name': 'Sign language detection',
        'entity': None,
        'run name': f'SLD-training-{run}'
    }

class trainer:
    def __init__(    
        self,
        config: TRAINING_CONFIG,
        dataset: Dataset,
        model: nn.Module
    ):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.optimizer = config.MAIN['optimizer_class'](
            model.parameters(), 
            **config.MAIN['optimizer_config']
        )

    def _get_dataloaders(self):
        ...
    def train(self):
        ...
    def validate(self):
        ...
    def log(self):
        ...