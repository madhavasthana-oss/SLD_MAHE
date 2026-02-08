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
        'max_grad_norm' : 1,
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
        self.train_dl, self.val_dl, self.test_dl = self._get_dataloaders()

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
        val_len = splits[1] * len(self.dataset)
        test_len = len(self.dataset) - train_len - val_len
        train_ds, val_ds, test_ds = random_split(
            self.dataset, [train_len, val_len, test_len]
        )
        print(
            f'<<<< dataset splits have been made >>>>'
            f'<<<< train dataset length -> {train_len} >>>>'
            f'<<<< validation dataset length -> {val_len} >>>>'
            f'<<<< test dataset length -> {test_len} >>>>'
        )
        train_dl = DataLoader(
            train_ds,
            **self.config['DATASET']['dataset_config']
        )
        val_dl = DataLoader(
            val_ds,
            **self.config['DATASET']['dataset_config']
        )
        test_dl = DataLoader(
            test_ds,
            **self.config['DATASET']['dataset_config'],

        )

        return train_dl, val_dl, test_dl

    def train(self):
        '''main training loop, call every epoch'''
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        grad_norm_sum = 0
        clip_count = 0
        max_norm = self.config.MAIN['max_grad_norm']
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_dl, desc = 'Batch')):
            # move inputs to device, this is essential to ensure fast training
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            # main sequence, borrowed directly from the example in torch's sgd source
            self.optimizer.zero_grad()
            outs = self.model(inputs)
            loss = self.loss_fn(outs, targets)
            loss.backward()

            # gradient norm clipping
            if self.config.MAIN['max_grad_norm'] is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_norm
                )
                # update count
                grad_norm_sum += total_norm.item()
                if total_norm.item() > max_norm:
                    clip_count = 1

            self.optimizer.step()   

            # begin logging of loss and other metrics
            train_loss += loss.item()

            # calculate accuracy
            preds = outs.argmax(1)
            total += targets.size(0)
            correct += (preds).eq(targets).sum().item()
        num_batches = len(self.train_dl)
        train_metrics = {
            'train_loss': train_loss / num_batches,
            'train_acc': (correct/total) * 100.
        }

        if max_norm is not None:
            train_metrics['mean_grad_norm'] = grad_norm_sum / num_batches
            train_metrics['gradient_clipping_ratio'] = clip_count / num_batches
        
        return train_metrics
    
    def validate(self):
        '''call after training at config.['MAIN']['validate_every'] frequency'''
        self.model.eval()
        val_loss = 0
        val_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.val_dl, dexc = 'batch')):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                val_loss += loss.item()
                total += targets.size()
                predictions = outputs.argmax(1)
                val_correct += predictions.eq(targets).sum().item()

        num_batches = len(self.val_dl)        
        val_accuracy = 100. * (val_correct/total)
        loss = val_loss / num_batches

        return {
            'val_loss':loss,
            'val_acc':val_accuracy
        }

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