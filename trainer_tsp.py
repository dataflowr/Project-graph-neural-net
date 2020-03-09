import time
import torch
import numpy as np

def train_tsp(train_loader,model,criterion,optimizer,
                logger,device,epoch,eval_score=None,print_freq=100):
    model.train()
    meters = logger.reset_meters('train')
    meters_params = logger.reset_meters('hyperparams')
    meters_params['learning_rate'].update(optimizer.param_groups[0]['lr'])
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        batch_size = input.shape[0]
        num_nodes = input.shape[1]
        # measure data loading time
        meters['data_time'].update(time.time() - end, n=batch_size)

        input = input.to(device)
        target = target.to(device)
        output = model(input)
        #print('output',output.shape,target.shape)
        

        loss = criterion(output,target)#.view(-1,num_nodes),target.view(-1))
        meters['loss'].update(loss.data.item(), n=1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        meters['batch_time'].update(time.time() - end, n=batch_size)
        end = time.time()

    
        if i % print_freq == 0:
            if eval_score is not None:
                np_out = output.cpu().detach().numpy()
                np_pred = np.argsort(-np_out, axis=2)[:,:,:2]
                np_pred = [np.asarray([[i,j] if i<j else [j,i] for [i,j] in t]) for t in np_pred]
                np_target = target.cpu().detach().numpy()
                np_target = np.argsort(-np_target, axis=2)[:,:,:2]
                np_target = [np.asarray([[i,j] if i<j else [j,i] for [i,j] in t]) for t in np_target]
                np_input = input[:,:,:,1].cpu().detach().numpy()
                #print(np_pred,np_target.shape)
                acc_max, n, bs = eval_score(np_pred,np_target)#,np_input)
                #print(acc_max, n, bs)
                meters['acc_max'].update(acc_max,(n*2)*bs)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr.val:.2e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc_Max {acc_max.avg:.3f} ({acc_max.val:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=meters['batch_time'],
                   data_time=meters['data_time'], lr=meters_params['learning_rate'], loss=meters['loss'], acc_max=meters['acc_max']))

    logger.log_meters('train', n=epoch)
    logger.log_meters('hyperparams', n=epoch)


def val_tsp(val_loader,model,criterion,
                logger,device,epoch,eval_score=None,print_freq=10):
    model.eval()
    meters = logger.reset_meters('val')

    for i, (input, target) in enumerate(val_loader):
        num_nodes = input.shape[1]
        input = input.to(device)
        target = target.to(device)
        output = model(input)

        loss = criterion(output,target)#.view(-1,num_nodes),target.view(-1))
        meters['loss'].update(loss.data.item(), n=1)
    
        if eval_score is not None:
            np_out = output.cpu().detach().numpy()
            np_pred = np.argsort(-np_out, axis=2)[:,:,:2]
            np_pred = [np.asarray([[i,j] if i<j else [j,i] for [i,j] in t]) for t in np_pred]
            np_target = target.cpu().detach().numpy()
            np_target = np.argsort(-np_target, axis=2)[:,:,:2]
            np_target = [np.asarray([[i,j] if i<j else [j,i] for [i,j] in t]) for t in np_target]
            np_input = input[:,:,:,1].cpu().detach().numpy()
                #print(np_out.shape)
            acc, n, bs = eval_score(np_pred, np_target)#,np_input)
                #print(acc_max, n, bs)
            meters['acc_la'].update(acc,(n*2)*bs)
        if i % print_freq == 0:
            print('Validation set, epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.avg:.3f} ({acc.val:.3f})'.format(
                    epoch, i, len(val_loader), loss=meters['loss'], acc=meters['acc_la']))

    logger.log_meters('val', n=epoch)
    return acc