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
    device: str = 'cuda' if torch.cuda.is_available else 'cpu'
    MAIN: Dict = {
        'loss_fn': nn.CrossEntropyLoss,
        'num_epochs': 200,
        'warmup_epochs': 10,
        'validate_every': 1,
        'optimizer_class': SGD(           
        ),
        'optimizer_config':{
            'lr':1e-2,
            'momentum':0.9,
            'nesterov':True,
            'dampening':False,
            'weight_decay':0.0005
        },
        'scheduler_class': CosineAnnealingLR,
        'scheduler_config':{
            'T_max':200,
            'eta_Min': 1e-4
        }
    }
    DATASET: Dict = {
        'batch_size': 512,
        'num_workers': 2,
    }
    WANDB: Dict = {
        'project_name': 'Sign language detection',
        'entity': None,
        'run_name': f'SLD-training-{run}'
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
        self.scheduler = config.MAIN['scheduler_class'](
            self.optimizer, 
            **self.config.MAIN['scheduler_config']
        )

    def _get_dataloaders(self):
        '''handles dataloading and seeding'''
        ...
    def train(self):
        '''main training loop, call every epoch'''
        ...
    def validate(self):
        '''call after training at config.['MAIN']['validate_every'] frequency'''
        ...
    def log(self):
        '''call in train, validate and test to log metrics'''
        ...
    def test(self):
        '''call after training is finished to test'''
        ...
    def fit(self):
        '''
        calls the following methods
            -> train
            -> validate
            for the specified number of epochs
        and finally 
        '''