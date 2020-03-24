from toolbox import metrics
import time

def train_triplet(train_loader,model,criterion,optimizer,
                logger,device,epoch,eval_score=None,print_freq=100):
    model.train()
    logger.reset_meters('train')
    logger.reset_meters('hyperparams')
    learning_rate = optimizer.param_groups[0]['lr']
    logger.update_value_meter('hyperparams', 'learning_rate', learning_rate)
    end = time.time()

    for i, (input1, input2) in enumerate(train_loader):
        batch_size = input1.shape[0]
        # measure data loading time
        logger.update_meter('train', 'data_time', time.time() - end, n=batch_size)

        input1 = input1.to(device)
        input2 = input2.to(device)
        output = model(input1,input2)

        loss = criterion(output)
        logger.update_meter('train', 'loss', loss.data.item(), n=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        logger.update_meter('train', 'batch_time', time.time() - end, n=batch_size)
        end = time.time()

    
        if i % print_freq == 0:
            if eval_score is not None:
                #print(np_out.shape)
                acc_max, n, bs = eval_score(output)
                #print(acc_max, n, bs)
                logger.update_meter('train', 'acc_max', acc_max, n=n*bs)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc_Max {acc_max.avg:.3f} ({acc_max.val:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=logger.get_meter('train', 'batch_time'),
                   data_time=logger.get_meter('train', 'data_time'), lr=learning_rate,
                   loss=logger.get_meter('train', 'loss'), acc_max=logger.get_meter('train', 'acc_max')))

    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


def val_triplet(val_loader,model,criterion,
                logger,device,epoch,eval_score=None,print_freq=10):
    model.eval()
    logger.reset_meters('val')

    for i, (input1, input2) in enumerate(val_loader):
        input1 = input1.to(device)
        input2 = input2.to(device)
        output = model(input1,input2)

        loss = criterion(output)
        logger.update_meter('val', 'loss', loss.data.item(), n=1)
    
        if eval_score is not None:
            acc_la, n, bs = eval_score(output)
            logger.update_meter('val', 'acc_la', acc_la, n=n*bs)
        if i % print_freq == 0:
            print('Validation set, epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.avg:.3f} ({acc.val:.3f})'.format(
                    epoch, i, len(val_loader), loss=logger.get_meter('val', 'loss'),
                    acc=logger.get_meter('val', 'acc_la')))

    logger.log_meters('val', n=epoch)
    return acc_la
