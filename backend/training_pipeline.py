from imports import *


class CosineAnnealingWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.

    Args:
        optimizer     : Wrapped optimizer.
        warmup_epochs : Number of epochs for linear warmup.
        T_max         : Total number of training epochs.
        eta_min       : Minimum learning rate (default: 0).
        last_epoch    : The index of last epoch (default: -1).
    """
    def __init__(self, optimizer, warmup_epochs, T_max, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max         = T_max
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.T_max - self.warmup_epochs
            )
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


@dataclass
class TRAINING_CONFIG:
    """
    Central config for the full training pipeline.

    MAIN     — loss, epochs, optimizer, scheduler, grad clipping, early stopping
    DATASET  — splits, dataloader configs
    WANDB    — project / run metadata
    """
    seed: int = 42
    run:  int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    MAIN: Dict = field(default_factory=lambda: {
        'loss_fn':                 nn.CrossEntropyLoss(),
        'num_epochs':              200,
        'validate_every':          1,
        'max_grad_norm':           1.0,
        'early_stopping_patience': 20,
        'optimizer_class':         SGD,
        'optimizer_config': {
            'lr':           1e-2,
            'momentum':     0.9,
            'nesterov':     True,
            'dampening':    0,
            'weight_decay': 0.0005
        },
        'scheduler_class':  CosineAnnealingWithWarmup,
        'scheduler_config': {
            'warmup_epochs': 10,
            'T_max':         200,
            'eta_min':       1e-4
        }
    })

    checkpoint:       str = 'BEST_MODEL.pth'
    final_checkpoint: str = 'FINAL_MODEL.pth'

    DATASET: Dict = field(default_factory=lambda: {
        'root_dir': '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train',
        'splits':   [0.7, 0.2, 0.1],
        'train_config': {
            'batch_size':  128,
            'num_workers': 2,
            'shuffle':     True,
            'drop_last':   False
        },
        'eval_config': {
            'batch_size':  128,
            'num_workers': 2,
            'shuffle':     False,
            'drop_last':   False
        }
    })

    WANDB: Dict = field(default_factory=lambda: {
        'project': 'Sign language detection',
        'entity':  None,
        'name':    None
    })


class trainer:
    def __init__(
        self,
        config:  TRAINING_CONFIG,
        dataset: Dataset,
        model:   nn.Module
    ):
        self.config    = config
        self.dataset   = dataset
        self.model     = model.to(config.device)
        self.optimizer = config.MAIN['optimizer_class'](
            model.parameters(),
            **config.MAIN['optimizer_config']
        )
        self.scheduler = config.MAIN['scheduler_class'](
            self.optimizer,
            **config.MAIN['scheduler_config']
        )
        self.loss_fn = config.MAIN['loss_fn']

        self.train_dl, self.val_dl, self.test_dl = self._get_dataloaders()

        self.best_accuracy    = 0.0
        self.patience_counter = 0
        self.epoch_pbar       = None
        self._best_model_saved = False
        self.global_step       = 0

    # ── Seeding ───────────────────────────────────────────────────────────────

    def __seed__(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # ── Dataloaders ───────────────────────────────────────────────────────────

    def _get_dataloaders(self):
        """
        Split the dataset and apply separate transforms to train vs eval splits.
        Train subset gets augmentation; val/test get clean eval transform.
        """
        self.__seed__()

        splits    = self.config.DATASET['splits']
        total     = len(self.dataset)
        train_len = int(splits[0] * total)
        val_len   = int(splits[1] * total)
        test_len  = total - train_len - val_len

        train_subset, val_subset, test_subset = random_split(
            self.dataset, [train_len, val_len, test_len]
        )

        # ✅ FIX: wrap the Subset objects themselves, don't overwrite .dataset
        # Previously: train_ds.dataset = _TransformSubset(train_ds, ...) which
        # caused the Subset's indices to point into a wrongly-sized dataset,
        # triggering IndexError: list index out of range.
        train_ds = _TransformSubset(train_subset, train_transform)
        val_ds   = _TransformSubset(val_subset,   eval_transform)
        test_ds  = _TransformSubset(test_subset,  eval_transform)

        print(
            f'<<<< Dataset splits >>>>\n'
            f'  train : {train_len}\n'
            f'  val   : {val_len}\n'
            f'  test  : {test_len}'
        )

        train_dl = DataLoader(train_ds, **self.config.DATASET['train_config'])
        val_dl   = DataLoader(val_ds,   **self.config.DATASET['eval_config'])
        test_dl  = DataLoader(test_ds,  **self.config.DATASET['eval_config'])

        return train_dl, val_dl, test_dl

    # ── Train loop ────────────────────────────────────────────────────────────

    def train(self):
        self.model.train()
        train_loss    = 0.0
        correct       = 0
        total         = 0
        grad_norm_sum = 0.0
        clip_count    = 0
        max_norm      = self.config.MAIN['max_grad_norm']
        num_batches = len(self.train_dl)
        batch_pbar  = tqdm(
            total=num_batches,
            desc='  train',
            position=1,
            leave=False,
            ncols=100,
            file=sys.stdout,
            dynamic_ncols=False
        )

        for batch_idx, (inputs, targets) in enumerate(self.train_dl):
            inputs  = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            self.optimizer.zero_grad()
            outs = self.model(inputs)
            loss = self.loss_fn(outs, targets)
            loss.backward()

            if max_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_norm
                )
                grad_norm_sum += total_norm.item()
                if total_norm.item() > max_norm:
                    clip_count += 1

            self.optimizer.step()

            train_loss += loss.item()
            preds       = outs.argmax(1)
            total      += targets.size(0)
            correct    += preds.eq(targets).sum().item()

            batch_log = {
                'batch/train_loss': loss.item(),
                'batch/train_acc':  100. * correct / total,
            }
            if max_norm is not None:
                batch_log['batch/grad_norm'] = total_norm.item()
            wandb.log(batch_log, step=self.global_step)
            self.global_step += 1

            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc':  f'{100. * correct / total:.2f}%'
            })
            batch_pbar.update(1)

        batch_pbar.close()

        train_metrics = {
            'train_loss': train_loss / num_batches,
            'train_acc':  100. * correct / total
        }

        if max_norm is not None:
            train_metrics['mean_grad_norm']          = grad_norm_sum / num_batches
            train_metrics['gradient_clipping_ratio'] = clip_count / num_batches

        return train_metrics

    # ── Eval loop ─────────────────────────────────────────────────────────────

    def eval(self, mode: str = 'validate'):
        self.model.eval()
        val_loss    = 0.0
        val_correct = 0
        total       = 0

        if mode == 'validate':
            dl = self.val_dl
        elif mode == 'test':
            dl = self.test_dl
        else:
            raise ValueError("mode must be 'validate' or 'test'")

        num_batches = len(dl)
        batch_pbar  = tqdm(
            total=num_batches,
            desc=f'  {mode}',
            position=1,
            leave=False,
            ncols=100,
            file=sys.stdout,
            dynamic_ncols=False
        )

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dl):
                inputs  = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                outputs = self.model(inputs)
                loss    = self.loss_fn(outputs, targets)

                val_loss    += loss.item()
                total       += targets.size(0)
                predictions  = outputs.argmax(1)
                val_correct += predictions.eq(targets).sum().item()

                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc':  f'{100. * val_correct / total:.2f}%'
                })
                batch_pbar.update(1)

        batch_pbar.close()

        prefix = 'val' if mode == 'validate' else 'test'
        return {
            f'{prefix}_loss': val_loss / len(dl),
            f'{prefix}_acc':  100. * val_correct / total
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy':        self.best_accuracy,
            'config':               self.config
        }
        if is_best:
            torch.save(checkpoint, self.config.checkpoint)
            self._best_model_saved = True
            self.epoch_pbar.write(
                f'<<<< New best model saved — val_acc: {self.best_accuracy:.2f}% >>>>'
            )
        torch.save(checkpoint, self.config.final_checkpoint)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        return checkpoint['epoch']

    # ── Main fit ──────────────────────────────────────────────────────────────

    def fit(self):
        """
        Full training loop:
          train → validate → early stopping → test on best checkpoint
        """
        wandb.init(
            **self.config.WANDB,
            config={
                'seed': self.config.seed,
                'run':  self.config.run,
                **self.config.MAIN,
                'train_config': self.config.DATASET['train_config'],
                'eval_config':  self.config.DATASET['eval_config']
            }
        )

        num_epochs              = self.config.MAIN['num_epochs']
        validate_every          = self.config.MAIN['validate_every']
        early_stopping_patience = self.config.MAIN['early_stopping_patience']

        print(f'\n<<<< Starting training for {num_epochs} epochs >>>>\n')
        self.epoch_pbar = tqdm(range(num_epochs), desc='Epochs', position=0, leave=True, ncols=100, file=sys.stdout, dynamic_ncols=False)

        last_epoch = 0
        for epoch in self.epoch_pbar:
            last_epoch    = epoch
            train_metrics = self.train()

            if (epoch + 1) % validate_every == 0:
                val_metrics  = self.eval('validate')
                self.scheduler.step()
                lr           = self.optimizer.param_groups[0]['lr']
                current_acc  = val_metrics['val_acc']

                wandb.log({'epoch': epoch, 'lr': lr, **train_metrics, **val_metrics}, step=self.global_step)

                self.epoch_pbar.set_postfix({
                    'epoch':      f'{epoch + 1}',
                    'train_loss': f'{train_metrics["train_loss"]:.4f}',
                    'train_acc':  f'{train_metrics["train_acc"]:.2f}%',
                    'val_loss':   f'{val_metrics["val_loss"]:.4f}',
                    'val_acc':    f'{val_metrics["val_acc"]:.2f}%',
                    'lr':         f'{lr:.6f}'
                })

                if current_acc > self.best_accuracy:
                    self.best_accuracy    = current_acc
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= early_stopping_patience:
                    self.epoch_pbar.write(
                        f'\n<<<< Early stopping at epoch {epoch + 1} | '
                        f'best val_acc: {self.best_accuracy:.2f}% >>>>'
                    )
                    break
            else:
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]['lr']
                self.epoch_pbar.set_postfix({
                    'train_loss': f'{train_metrics["train_loss"]:.4f}',
                    'train_acc':  f'{train_metrics["train_acc"]:.2f}%',
                    'lr':         f'{lr:.6f}'
                })
                wandb.log({'epoch': epoch, 'lr': lr, **train_metrics}, step=self.global_step)

        self.epoch_pbar.close()

        self.save_checkpoint(last_epoch, is_best=False)

        print(f'\n<<<< Training complete >>>>')
        if self._best_model_saved:
            self.epoch_pbar.write('<<<< Loading best checkpoint for final evaluation >>>>')
            self.load_checkpoint(self.config.checkpoint)
        else:
            self.epoch_pbar.write('<<<< Warning: no best checkpoint saved, evaluating final model >>>>')

        test_metrics = self.eval('test')

        wandb.log({
            'test_loss': test_metrics['test_loss'],
            'test_acc':  test_metrics['test_acc']
        })

        print(
            f'\n<<<< Final Test Results >>>>\n'
            f'  test_loss : {test_metrics["test_loss"]:.4f}\n'
            f'  test_acc  : {test_metrics["test_acc"]:.2f}%'
        )

        wandb.finish()

        return {
            'best_val_acc': self.best_accuracy,
            'test_acc':     test_metrics['test_acc'],
            'test_loss':    test_metrics['test_loss']
        }


# ── Helper: wrapper to apply a different transform to a Subset ────────────────

class _TransformSubset(Dataset):
    """
    Wraps a Subset so it can carry its own transform,
    independent of the parent dataset's transform.
    """
    def __init__(self, subset: Dataset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # subset[idx] returns a PIL image if the parent dataset has transform=None
        # apply our split-specific transform here
        if isinstance(image, torch.Tensor):
            # shouldn't happen when parent dataset has transform=None,
            # but guard anyway
            return image, label
        return self.transform(image), label


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',   type=str, required=True)
    parser.add_argument('--save_to',    type=str, required=True)
    parser.add_argument('--epochs',     type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    config = TRAINING_CONFIG()
    config.DATASET['root_dir']                        = args.root_dir
    config.MAIN['num_epochs']                         = args.epochs
    config.MAIN['scheduler_config']['T_max']          = args.epochs
    config.DATASET['train_config']['batch_size']      = args.batch_size
    config.DATASET['eval_config']['batch_size']       = args.batch_size
    config.checkpoint                                 = os.path.join(args.save_to, 'BEST_MODEL.pth')
    config.final_checkpoint                           = os.path.join(args.save_to, 'FINAL_MODEL.pth')

    os.makedirs(args.save_to, exist_ok=True)

    dataset = ASLDataset(root_dir=config.DATASET['root_dir'], transform=None)
    model   = SLD(num_classes=len(dataset.classes))

    trainer_instance = trainer(config, dataset, model)
    results = trainer_instance.fit()
    print(results)