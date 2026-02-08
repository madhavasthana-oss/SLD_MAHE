from imports import * 

class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for linear warmup
        max_epochs: Total number of training epochs
        eta_min: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]


@dataclass
class TRAINING_CONFIG:
    '''
    This is the TRAINING_CONFIG dataclass
    it will be used to define the following
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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    MAIN: Dict = field(
        default_factory=lambda:{
        'loss_fn': nn.CrossEntropyLoss(),
        'num_epochs': 200,
        'validate_every': 1,
        'max_grad_norm': 1,
        'early_stopping_patience': 20,
        'optimizer_class': SGD,
        'optimizer_config':{
            'lr': 1e-2,
            'momentum': 0.9,
            'nesterov': True,
            'dampening': False,
            'weight_decay': 0.0005
        },
        'scheduler_class': CosineAnnealingWithWarmup,
        'scheduler_config':{
            'warmup_epochs': 10,
            'T_max': 200,
            'eta_min': 1e-4
        }
    })

    checkpoint: str = 'BEST_MODEL.pth'
    final_checkpoint: str = 'FINAL_MODEL.pth'

    DATASET: Dict = field(default_factory=lambda:{
        'splits': [0.7, 0.2, 0.1],
        'train_config': {
            'batch_size': 512,
            'num_workers': 2,
            'shuffle': True,
            'drop_last': True
        },
        'eval_config': {
            'batch_size': 512,
            'num_workers': 2,
            'shuffle': False,
            'drop_last': False
        }
    })

    WANDB: Dict = field(default_factory=lambda:{
        'project': 'Sign language detection',
        'entity': None,
        'name': None
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
        self.model = model.to(config.device)
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
        
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.epoch_pbar = None

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
        train_len = int(splits[0] * len(self.dataset))
        val_len = int(splits[1] * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len
        train_ds, val_ds, test_ds = random_split(
            self.dataset, [train_len, val_len, test_len]
        )
        print(
            f'<<<< dataset splits have been made >>>>\n'
            f'<<<< train dataset length -> {train_len} >>>>\n'
            f'<<<< validation dataset length -> {val_len} >>>>\n'
            f'<<<< test dataset length -> {test_len} >>>>'
        )
        train_dl = DataLoader(
            train_ds,
            **self.config.DATASET['train_config']
        )
        val_dl = DataLoader(
            val_ds,
            **self.config.DATASET['eval_config']
        )
        test_dl = DataLoader(
            test_ds,
            **self.config.DATASET['eval_config']
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
        
        batch_pbar = tqdm(
            enumerate(self.train_dl), 
            total=len(self.train_dl),
            desc='Training',
            leave=False,
            position=1
        )
        
        for batch_idx, (inputs, targets) in batch_pbar:
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            self.optimizer.zero_grad()
            outs = self.model(inputs)
            loss = self.loss_fn(outs, targets)
            loss.backward()

            if self.config.MAIN['max_grad_norm'] is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_norm
                )
                grad_norm_sum += total_norm.item()
                if total_norm.item() > max_norm:
                    clip_count += 1

            self.optimizer.step()   
            
            train_loss += loss.item()

            preds = outs.argmax(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            
            current_acc = (correct / total) * 100.
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
        batch_pbar.close()
        
        num_batches = len(self.train_dl)
        train_metrics = {
            'train_loss': train_loss / num_batches,
            'train_acc': (correct / total) * 100.
        }

        if max_norm is not None:
            train_metrics['mean_grad_norm'] = grad_norm_sum / num_batches
            train_metrics['gradient_clipping_ratio'] = clip_count / num_batches
        
        return train_metrics
    
    def eval(self, mode="validate"):
        '''call after training at config.MAIN.validate_every frequency'''
        self.model.eval()
        val_loss = 0
        val_correct = 0
        total = 0
        
        if mode == 'validate':
            dl = self.val_dl
        elif mode == 'test':
            dl = self.test_dl
        else:
            raise ValueError("invalid mode, please select between 'validate' or 'test'")
        
        batch_pbar = tqdm(
            enumerate(dl),
            total=len(dl),
            desc=f'{mode.capitalize()}',
            leave=False,
            position=1
        )
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in batch_pbar:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                val_loss += loss.item()
                total += targets.size(0)
                predictions = outputs.argmax(1)
                val_correct += predictions.eq(targets).sum().item()
                
                current_acc = (val_correct / total) * 100.
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        batch_pbar.close()

        num_batches = len(dl)        
        val_accuracy = 100. * (val_correct / total)
        loss = val_loss / num_batches

        prefix = 'val' if mode == 'validate' else 'test'
        return {
            f'{prefix}_loss': loss,
            f'{prefix}_acc': val_accuracy
        }
        
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, self.config.checkpoint)
            tqdm.write(f'<<<< Saved best model with accuracy: {self.best_accuracy:.2f}% >>>>')
        
        torch.save(checkpoint, self.config.final_checkpoint)

    def load_checkpoint(self, checkpoint_path):
        '''load a saved checkpoint to resume training or for inference'''
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        return checkpoint['epoch']

    def fit(self):
        '''
        calls the following methods
            -> train
            -> validate
            for the specified number of epochs
        implements early stopping and best model checkpointing
        finally evaluates on test set
        '''
        wandb.init(
           **self.config.WANDB,
           config={
               'seed': self.config.seed,
               'run': self.config.run,
               **self.config.MAIN,
               'train_config': self.config.DATASET['train_config'],
               'eval_config': self.config.DATASET['eval_config']
           }
        )
        
        num_epochs = self.config.MAIN['num_epochs']
        validate_every = self.config.MAIN['validate_every']
        early_stopping_patience = self.config.MAIN['early_stopping_patience']
        
        tqdm.write(f'\n<<<< Starting training for {num_epochs} epochs >>>>\n')
        
        self.epoch_pbar = tqdm(range(num_epochs), desc='Epochs', position=0)
        
        for epoch in self.epoch_pbar:
            train_metrics = self.train()
            
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.eval('validate')
                
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                
                current_acc = val_metrics['val_acc']
                
                wandb.log({
                    'epoch': epoch,
                    'lr': lr,
                    **train_metrics,
                    **val_metrics
                })
                
                self.epoch_pbar.set_postfix({
                    'train_loss': f'{train_metrics["train_loss"]:.4f}',
                    'train_acc': f'{train_metrics["train_acc"]:.2f}%',
                    'val_loss': f'{val_metrics["val_loss"]:.4f}',
                    'val_acc': f'{val_metrics["val_acc"]:.2f}%',
                    'lr': f'{lr:.6f}'
                })
                
                tqdm.write(
                    f'Epoch {epoch+1}/{num_epochs} | '
                    f'Train Loss: {train_metrics["train_loss"]:.4f} | '
                    f'Train Acc: {train_metrics["train_acc"]:.2f}% | '
                    f'Val Loss: {val_metrics["val_loss"]:.4f} | '
                    f'Val Acc: {val_metrics["val_acc"]:.2f}% | '
                    f'LR: {lr:.6f}'
                )
                
                if current_acc > self.best_accuracy:
                    self.best_accuracy = current_acc
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= early_stopping_patience:
                    tqdm.write(f'\n<<<< Early stopping triggered after {epoch+1} epochs >>>>')
                    tqdm.write(f'<<<< Best validation accuracy: {self.best_accuracy:.2f}% >>>>')
                    break
            else:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                
                self.epoch_pbar.set_postfix({
                    'train_loss': f'{train_metrics["train_loss"]:.4f}',
                    'train_acc': f'{train_metrics["train_acc"]:.2f}%',
                    'lr': f'{lr:.6f}'
                })
                
                wandb.log({
                    'epoch': epoch,
                    'lr': lr,
                    **train_metrics
                })
        
        self.epoch_pbar.close()
        
        self.save_checkpoint(epoch, is_best=False)
        
        tqdm.write(f'\n<<<< Training completed >>>>')
        tqdm.write(f'<<<< Loading best model for final evaluation >>>>')
        self.load_checkpoint(self.config.checkpoint)
        
        test_metrics = self.eval('test')
        
        wandb.log({
            'test_loss': test_metrics['test_loss'],
            'test_acc': test_metrics['test_acc']
        })
        
        tqdm.write(f'\n<<<< Final Test Results >>>>')
        tqdm.write(f'<<<< Test Loss: {test_metrics["test_loss"]:.4f} >>>>')
        tqdm.write(f'<<<< Test Acc: {test_metrics["test_acc"]:.2f}% >>>>')
        
        wandb.finish()
        
        return {
            'best_val_acc': self.best_accuracy,
            'test_acc': test_metrics['test_acc'],
            'test_loss': test_metrics['test_loss']
        }