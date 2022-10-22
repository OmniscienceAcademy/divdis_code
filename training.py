from itertools import cycle, chain
from defaults import *
from evaluation import *
from losses import *
from evaluation import *
from utils import *
from datasets import Container


def train_epoch_joint(models: List[nn.Module], labeled, unlabeled, max_batches, labeled_criterion: nn.Module,
                      unlabeled_criterion: nn.Module, optim: torch.optim.Optimizer, labeled_scale=1.0,
                      distinct_scale=1.0, class_certainty_scale=1.0, loss_type=0, smooth=False, dynamic=False,
                      device='cpu'):
    # track labeled and unlabeled losses
    cum_labeled, cum_distinct, cum_class_certainty, cum_loss = 0., 0., 0., 0.

    labeled   = cycle(iter(labeled))
    unlabeled = cycle(iter(unlabeled)) 
    i = 0

    while i < 2:
        labeled_batch, labels, _ = next(labeled) # confusingly, "targets" just means "labels"
        unlabeled_batch, _, hidden_labels = next(unlabeled) # unlab_targets is a tensor of labels for each feature

        log_img = ((i + 1) % image_log_frequency == 0)
        if log_img: print('Logging images...')
        # calculate labeled loss
        labeled_loss = get_labeled_loss(models, labeled_criterion, labeled_batch, labels, device=device, log=log_img)
        # calculate unlabeled loss
        distinct_loss, class_certainty_loss = get_unlabeled_losses(models, unlabeled_criterion, unlabeled_batch, hidden_labels,
                                                                   loss_type=loss_type, device=device, log=log_img,
                                                                   smooth=smooth,
                                                                   dynamic=dynamic)

        # weighted sum of labeled and unlabeled losses
        loss = labeled_scale * labeled_loss + distinct_scale * distinct_loss + class_certainty_scale * class_certainty_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        print('Batch {}/{} Loss: {}'.format(i, max_batches, loss))

        # track losses
        cum_labeled += labeled_loss.detach().cpu()
        cum_distinct += distinct_loss.detach().cpu()
        cum_class_certainty += class_certainty_loss#.detach().cpu() TODO readd this
        cum_loss += loss.detach().cpu()
        i += 1

    return cum_loss, cum_labeled, cum_distinct, cum_class_certainty


class DistinctTrainer():

    def __init__(self, models: List[nn.Module], name: str, container: Container,
                 labeled_criterion: nn.Module, unlabeled_criterion: nn.Module, optim=torch.optim.Adam,
                 lr=1e-3, weight_decay=1e-3, labeled_scale=1, distinct_scale=1, class_certainty_scale=1,
                 loss_type=0, lpft=False, device='cpu', log=True, test_on_train=False, smooth=False,
                 dynamic=False, saliency='none', saliency_step=-1, saliency_ksize=10, limit=-1, **kwargs):
        print('Building trainer...')

        self.loss_type = loss_type
        if loss_type == 1:
            print('Using divdis...')
        if loss_type == 2:
            print('Using correct feature labels...')


        self.lpft = lpft
        if lpft:
            print('Training with LP-FT...')
            self.lp_optim = optim(chain(*[model.class_heads.parameters() for model in models]), lr=lr[0], weight_decay=weight_decay)
            self.ft_optim = optim(chain(*[model.parameters() for model in models]), lr=lr[1], weight_decay=weight_decay)
        else:
            self.optim = optim(chain(*[model.parameters() for model in models]), lr=lr, weight_decay=weight_decay)

        self.log = log
        self.saliency = saliency
        self.saliency_ksize = saliency_ksize
        self.saliency_step = saliency_step
        print('Sending models to {}...'.format(device))
        self.models = [model.to(device) for model in models]

        if device == 'cuda':
            self.models = [nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))) for model in
                           models]

        self.limit = limit
        self.name = name
        self.container = container
        self.n_features = self.container.n_features
        self.test_on_train = test_on_train
        self.smooth = smooth
        self.dynamic = dynamic
        self.labeled_criterion = labeled_criterion
        self.unlabeled_criterion = unlabeled_criterion
        if self.unlabeled_criterion.reduction != 'none':
            raise ValueError("unlabeled criterion must have reduction = 'none")

        self.labeled_scale = labeled_scale
        self.distinct_scale = distinct_scale
        self.class_certainty_scale = class_certainty_scale

        self.device = device
        self.labeled = container.get_dataloader('labeled')
        self.unlabeled = container.get_dataloader('unlabeled')
        self.val = container.get_dataloader('val')
        self.test = container.get_dataloader('test')
        self.max_batches = max(len(self.labeled), len(self.unlabeled))
        self.labeled_batch_size, self.unlabeled_batch_size = container.batch_size['labeled'], container.batch_size['unlabeled']

    def evaluate(self, test=False):
        self.set_models_train(False)
        eval_mode = 'test' if test else 'val'

        data = self.test if test else self.val

        #labeled_accuracy = evaluate_source(self.models, source, device=self.device)
        diag_accuracy, head_performances, subset_predictions = evaluate_unlabeled(self.models, data, self.n_features,
                                                                                  device=self.device, get_subset_predictions=True)




        if len(self.models) > 1:
            n = len(self.models)
        else:
            if self.device == 'cuda':
                n = self.models[0].module.get_num_heads()
            else:
                n = self.models[0].get_num_heads()

        worst_head_loss = 1 - min(head_performances)
        best_head_loss = 1 - max(head_performances)

        print(f"worst {eval_mode} head accuracy = accuracy on the hard feature = {1-worst_head_loss}")
        print(f" best {eval_mode} head accuracy = accuracy on the easy feature = {1-best_head_loss}")

        self.set_models_train(True)

        return worst_head_loss

    def train(self, epochs):
        print('Training...')

        self.set_models_train(True)

        best_training_loss = 999999
        best_validation_loss = 999999

        if self.lpft:
            lp_epochs = epochs[0]
            epochs = sum(epochs) # total number of epochs
            print('Starting LP...')
            optim = self.lp_optim
        else:
            optim = self.optim

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs))
            l_scale, d_scale, cs_scale = self.labeled_scale, self.distinct_scale, self.class_certainty_scale

            if self.lpft and epoch == lp_epochs:
                print('Starting FT...')
                optim = self.ft_optim

            loss, labeled_loss, distinct_loss, class_certainty_loss = train_epoch_joint(self.models,
                                                                                        self.labeled,
                                                                                        self.unlabeled,
                                                                                        self.max_batches,
                                                                                        self.labeled_criterion,
                                                                                        self.unlabeled_criterion,
                                                                                        optim,
                                                                                        l_scale,
                                                                                        d_scale, cs_scale,
                                                                                        self.loss_type,
                                                                                        smooth=self.smooth,
                                                                                        dynamic=self.dynamic,
                                                                                        device=self.device)


            if loss < best_training_loss:
                best_training_loss = loss
                print('Lowest training loss so far!')

            wh_loss = self.evaluate()

            if wh_loss < best_validation_loss:
                best_validation_loss = wh_loss
                print('Lowest validation loss so far!')
                save_params(self.models, self.name)


    def set_models_train(self, train=True):
        for model in self.models:
            model.train(train)
