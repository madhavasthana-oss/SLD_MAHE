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
    MAIN: Dict = field(
        default_factory=lambda:{
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
    })
    DATASET: Dict = field(default_factory=lambda:{
        'splits': [0.7, 0.25, 0.05],
        'dataset_config': {
            'batch_size': 512,
            'num_workers': 2,
            'shuffle' : True
        }
    })
    WANDB: Dict = field(default_factory=lambda:{
        'project_name': 'Sign language detection',
        'entity': None
    })

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
        self.loss_fn = config.MAIN['loss_fn']
        self.train_loader, self.validation_loader, self.test_loader = self._get_dataloaders()

    def __seed__(self):
        '''Set random seeds for reproducibility'''
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_dataloaders(self):
        '''handles dataloading and seeding'''
        self.__seed__()
        splits = self.config.DATASET['splits']
        train_len = splits[0] * len(self.dataset)
        train_loader = DataLoader(
            self.dataset,
            **self.config['DATASET']['dataset_config']
        )

    def train(self, input):
        '''main training loop, call every epoch'''
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(input), output)
        loss.backward()


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