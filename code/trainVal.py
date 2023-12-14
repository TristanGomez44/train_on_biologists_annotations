import os
import sys
import glob

from shutil import copyfile
import gc
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import optuna
import sqlite3

from utils import _remove_no_annot
from args import ArgReader,str2bool,addInitArgs,addValArgs,init_post_hoc_arg,addLossTermArgs,addSalMetrArgs
import init_model
from loss import SupervisedLoss,SelfSuperVisedLoss,agregate_losses
import modelBuilder
import load_data
import metrics
import update
import utils 

def log_gradient_norms(exp_id,model_id,model,epoch,batch_idx,stat_list=["mean","std"]):

    grad_dict = {}

    named_layers = dict(model.named_modules())

    for layer_name in named_layers:
        layer = named_layers[layer_name]

        for param in ["weight","bias"]:
            if hasattr(layer,param):
                param_tensor = getattr(layer,param)
                if param_tensor is not None:
                    grad = param_tensor.grad.data.abs()

                    key = layer_name+"."+param
                    grad_dict[key] = {}

                    for stat in stat_list:
                        grad_dict[key][stat] = getattr(grad,stat)().item()

    param_names = sorted(grad_dict.keys())
    
    for stat in stat_list:
        csv_path = f"../results/{exp_id}/gradnorm_{stat}_{model_id}.csv"

        if not os.path.exists(csv_path):
            with open(csv_path,"w") as file:
                header = ["epoch","batch_idx"]+param_names
                header = ",".join(header)
                print(header,file=file)

        with open(csv_path,"a") as file:
            row = [str(epoch),str(batch_idx)]+[str(grad_dict[param_name][stat]) for param_name in grad_dict]
            row = ",".join(row)
            print(row,file=file)

def to_cuda(batch):
    for elem in batch:
        if type(elem) is torch.Tensor:
            elem = elem.cuda(non_blocking=True)
        elif type(elem) is dict:
            for key in elem:
                elem[key] = elem[key].cuda(non_blocking=True)
    return batch 

def _remove_excess_examples(tensor,end_ind):
    return tensor[:end_ind]

def remove_excess_examples(batch,accumulated_size,batch_size):
    if accumulated_size + batch[0].size(0) > batch_size:
        end_ind = max(batch_size-accumulated_size,2*torch.cuda.device_count())
        for elem in batch:
            if type(elem) is torch.Tensor:
                elem = _remove_excess_examples(elem,end_ind)
            elif type(elem) is dict:
                for key in elem:
                    elem[key] = _remove_excess_examples(elem[key],end_ind)
        accumulated_size = batch_size
    else:
        accumulated_size += batch[0].size(0)

    return batch,accumulated_size

def optim_step(model,optim,acc_nb):
    #Scale gradients (required when using gradient accumulation)
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= acc_nb
    optim.step()
    optim.zero_grad()
    accumulated_size = 0
    acc_nb = 0

    return model,optim,accumulated_size,acc_nb

def add_suff(dic,suff):
    new_dic = {}
    for key in dic:
        new_dic[key+suff] = dic[key]
    return new_dic

def inference(model,data1,data2,model_temp):
    student_dict = {}
    for i,data in enumerate([data1,data2]):
        student_dict_i = model(data) 
        student_dict_i = add_suff(student_dict_i,str(i+1))
        student_dict.update(student_dict_i)
    student_dict["temp"] = model_temp
    return student_dict

def teacher_inference(model,data1,data2,model_temp):
    teacher_dict = inference(model,data1,data2,model_temp)
    teacher_dict["center"] = model.center
    return teacher_dict

def compute_loss(loss_func,loss_args,backpropagate=True):
    loss_dic = loss_func(*loss_args)
    loss_dic = agregate_losses(loss_dic)
    loss = loss_dic["loss"]/len(list(loss_args[0].values())[0])
    if backpropagate:
        loss.backward()
    return loss_dic 

def self_supervised_step(model,batch,args,kwargs,is_train=True):
    data1,data2 = batch[0],batch[1]
    utils.save_image(data1,f"../vis/data1_{is_train}.png")
    utils.save_image(data2,f"../vis/data2_{is_train}.png")
    student_dict = inference(model,data1,data2,args.student_temp)
    with torch.no_grad():
        teacher_dict = inference(kwargs["teacher_net"],data1,data2,args.teach_temp)
        teacher_dict["center"] = kwargs["teacher_net"].center

    loss_dic = compute_loss(kwargs["loss_func"],[student_dict,teacher_dict],backpropagate=is_train)
    kwargs["teacher_net"] = update.update_teacher(kwargs["teacher_net"],model,args.teach_momentum)
    kwargs["teacher_net"].center = update.update_center(kwargs["teacher_net"].center,teacher_dict,args.teach_center_momentum)
    metDictSample = {}     
    output_dict = {"feat":student_dict["feat1"]}       
    return model,loss_dic,metDictSample,output_dict

def compute_norm(feat):
    return torch.sqrt(torch.pow(feat.sum(dim=1,keepdim=True),2))

def min_max_norm(tensor):
    tensor_min,tensor_max = tensor,tensor
    for i in [1,2,3]:
        tensor_min = tensor.min(dim=i,keepdim=True)[0]
        tensor_max = tensor.max(dim=i,keepdim=True)[0]
    return (tensor-tensor_min)/(tensor_max-tensor_min)

def crop_to_attention(data,output_dict,threshold=0.25):

    if not "attMaps" in output_dict:
        attMaps = compute_norm(output_dict["feat"])
    else:
        attMaps = output_dict["attMaps"].mean(dim=1,keepdim=True)

    attMaps = min_max_norm(attMaps)

    utils.save_image(torch.nn.functional.interpolate(attMaps,size=(256,256)),"../vis/data_attMaps.png")

    masks = attMaps>threshold

    print(masks.float().mean())

    orig_size = data.size()[2:]

    ratio = orig_size[-1]//attMaps.size(-1)

    cropped_img_list = []
    for k in range(len(masks)):

        salient_area_coords = torch.argwhere(masks[k,0]>0)

        min_i = salient_area_coords[:,0].min()*ratio
        max_i = salient_area_coords[:,0].max()*ratio
        min_j = salient_area_coords[:,1].min()*ratio
        max_j = salient_area_coords[:,1].max()*ratio

        print(min_i.item(),max_i.item(),min_j.item(),max_j.item())

        cropped_img = data[k:k+1,:,min_i:max_i,min_j:max_j]
        print(cropped_img.shape,orig_size)
        cropped_img = torch.nn.functional.interpolate(cropped_img,size=orig_size)
        cropped_img_list.append(cropped_img)

    cropped_img_batch = torch.cat(cropped_img_list,dim=0)

    return cropped_img_batch

def supervised_step(model,batch,kwargs,is_train=True,class_nb_dic=None):
    data, target_dic = batch[0], batch[1]

    output_dict = {}

    if kwargs["master_net"] is not None:
        with torch.no_grad():
            mast_output_dict = kwargs["master_net"](data)
            for key in mast_output_dict:
                if "output" in key:
                    output_dict["master_"+key] = mast_output_dict[key]

    output_dict.update(model(data))

    loss_dic = compute_loss(kwargs["loss_func"],[target_dic,output_dict],backpropagate=is_train)
    metDictSample = metrics.compute_metrics(target_dic,output_dict,class_nb_dic=class_nb_dic)
    return model,loss_dic,metDictSample,output_dict

def training_epoch(model, optim, loader, epoch, args, **kwargs):

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0

    accumulated_size = 0
    acc_nb = 0
    total_example_nb = 0 

    var_dic = {}
    for batch_idx, batch in enumerate(loader):
        optim.zero_grad()

        if batch_idx % args.log_interval == 0:
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        if args.cuda:
            batch = to_cuda(batch)

        batch,accumulated_size = remove_excess_examples(batch,accumulated_size,args.batch_size)

        acc_nb += 1
        total_example_nb += len(batch[0])

        if args.ssl:
            model,loss_dic,metDictSample,output_dict = self_supervised_step(model,batch,args,kwargs,is_train=True)
        else:
            model,loss_dic,metDictSample,output_dict = supervised_step(model,batch,kwargs,is_train=True,class_nb_dic=kwargs["class_nb_dic"])

        if args.log_gradient_norm_frequ is not None and batch_idx%args.log_gradient_norm_frequ==0:
            log_gradient_norms(args.exp_id,args.model_id,model,epoch,batch_idx)

        if accumulated_size == args.batch_size:
            model,optim,accumulated_size,acc_nb = optim_step(model,optim,acc_nb)

        # Metrics
        metDictSample = metrics.add_losses_to_dic(metDictSample,loss_dic)
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        var_dic = update.all_cat_var_dic(var_dic,output_dict,"train",args.save_output_during_validation)
            
        validBatch += 1

        if validBatch > 0 and args.debug:
            break
    
    if args.optuna:
        optuna_suff = "_trial"+str(args.trial_id)
    else:
        optuna_suff = ""

    writeSummaries(metrDict,total_example_nb,epoch, "train", args.model_id+optuna_suff, args.exp_id)

    return metrDict

def evaluation(model, loader, epoch, args, mode="val",**kwargs):

    model.eval()

    print("Epoch", epoch, " : {}".format(mode))

    metrDict = None
    validBatch = 0

    total_example_nb = 0  

    var_dic = {}
    for batch_idx, batch in enumerate(loader):

        if batch_idx % args.log_interval == 0:
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        if args.cuda:
            batch = to_cuda(batch)

        total_example_nb += len(batch[0])

        if args.ssl:
            model,loss_dic,metDictSample,output_dict = self_supervised_step(model,batch,args,kwargs,is_train=False)
        else:
            model,loss_dic,metDictSample,output_dict = supervised_step(model,batch,kwargs,is_train=False,class_nb_dic=kwargs["class_nb_dic"])

        # Metrics
        metDictSample = metrics.add_losses_to_dic(metDictSample,loss_dic)
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        var_dic = update.all_cat_var_dic(var_dic,output_dict,mode,args.save_output_during_validation)
            
        validBatch += 1

        if validBatch > 0 and args.debug:
            break

    if mode in ["test","val"]:
        update.save_variables(var_dic, args.exp_id, args.model_id, epoch, mode)

    if args.optuna:
        optuna_suff = "_trial"+str(args.trial_id)
    else:
        optuna_suff = ""

    writeSummaries(metrDict,total_example_nb, epoch, mode, args.model_id+optuna_suff, args.exp_id)

    if args.ssl:
        return metrDict["loss"]
    else:
        return metrDict["Accuracy_"+("EXP" if args.task_to_train == "all" else args.task_to_train)] 

def writeSummaries(metrDict,total_example_nb, epoch, mode, model_id, exp_id):

    for metric in metrDict.keys():
        metrDict[metric] /= total_example_nb

    header_list = ["epoch"]
    header_list += [metric.lower().replace(" ", "_") for metric in metrDict.keys()]
    header = ",".join(header_list)

    csv_path = f"../results/{exp_id}/metrics_{model_id}_{mode}.csv"

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as text_file:
           print(header, file=text_file) 

    with open(csv_path, "a") as text_file:
        print(epoch,file=text_file,end=",")
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]), file=text_file)

    return metrDict

def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=float, metavar='LR',
                                  help='learning rate')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                                  help='SGD momentum')
    argreader.parser.add_argument('--weight_decay', type=float, metavar='M',
                                  help='Weight decay')
    argreader.parser.add_argument('--use_scheduler', type=str2bool, metavar='M',
                                  help='To use a learning rate scheduler')
    
    argreader.parser.add_argument('--swa', type=str2bool, metavar='M',
                                  help='To run the swa/lr scheduler of the blastocyst dataset authors.') 
    
    argreader.parser.add_argument('--end_cosine_sched_epoch', type=int, metavar='M',
                                  help='Epoch at which cosine annealing end and lr becomes constant.')   
    argreader.parser.add_argument('--end_lr', type=float, metavar='M',
                                  help='Learning rate at the end of optimization.')   
    argreader.parser.add_argument('--warmup_lr', type=float, metavar='M',
                                  help='Initial lr during warmup.')       
    argreader.parser.add_argument('--warmup_epochs', type=int, metavar='M',
                                  help='Warmup length.')       
    argreader.parser.add_argument('--final_lr', type=float, metavar='M',
                                  help='Ending value for weight decay')

    argreader.parser.add_argument('--always_sched', type=str2bool, metavar='M',
                                  help='To always use a learning rate scheduler when optimizing hyper params')
    argreader.parser.add_argument('--sched_step_size', type=int, metavar='M',
                                  help='The number of epochs before reducing learning rate.')
    argreader.parser.add_argument('--sched_gamma', type=float, metavar='M',
                                  help='Multiplicative factor of learning rate decay')

    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                                  help='the optimizer to use (default: \'SGD\')')

    return argreader

def addRegressionArgs(argreader):
    argreader.parser.add_argument('--regression', type=str2bool, metavar='BOOL',
                                  help='To use self-supervised learning')
    argreader.parser.add_argument('--rank_weight', type=float, metavar='float',
                                  help='Weight of the rank loss term')
    argreader.parser.add_argument('--plcc_weight', type=float, metavar='float',
                                  help='Weight of the pearson linear correlation coeff. loss term.')
    return argreader
def addSSLArgs(argreader):

    argreader.parser.add_argument('--ssl', type=str2bool, metavar='BOOL',
                                  help='To use self-supervised learning')
    
    argreader.parser.add_argument('--ssl_data_augment', type=str2bool, metavar='BOOL',
                                  help='To use self-supervised learning')
    
    argreader.parser.add_argument('--only_center_plane', type=str2bool, metavar='BOOL',
                                  help='To only use the central focal plane.')

    argreader.parser.add_argument('--start_teach_temp', type=float, metavar='M',
                                  help='Starting temperature for the softmax of the teacher model.')
    
    argreader.parser.add_argument('--end_teach_temp', type=float, metavar='M',
                                  help='Ending temperature for the softmax of the teacher model.')
    
    argreader.parser.add_argument('--teach_temp_sched_epochs', type=int, metavar='M',
                                  help='Number of epochs over which the temperature of the softmax of the teacher model is increased.')
  
    argreader.parser.add_argument('--student_temp', type=float, metavar='M',
                                  help='Temperature for the softmax of the student model.')

    argreader.parser.add_argument('--start_teach_momentum', type=float, metavar='M',
                                  help='Starting momentum for the softmax of the teacher model.')

    argreader.parser.add_argument('--end_teach_momentum', type=float, metavar='M',
                                  help='Ending momentum for the softmax of the teacher model.')

    argreader.parser.add_argument('--teach_center_momentum', type=float, metavar='M',
                                  help='Momentum for the center update.')

    argreader.parser.add_argument('--start_weight_decay', type=float, metavar='M',
                                  help='Starting value for weight decay')
  
    argreader.parser.add_argument('--end_weight_decay', type=float, metavar='M',
                                  help='Ending value for weight decay')
    
    argreader.parser.add_argument('--ref_batch_size', type=int, metavar='M',
                                  help='Reference batch size to automatically compute the learning rate')

    argreader.parser.add_argument('--ref_lr', type=float, metavar='M',
                                  help='Learning rate at reference batch size.')

    return argreader

def run(args,trial):

    args.lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    args.optim = trial.suggest_categorical("optim", ["Adam", "AMSGrad", "SGD","AdamW"])

    if args.max_batch_size <= 12:
        minBS = 4
    else:
        minBS = 12

    args.batch_size = trial.suggest_int("batch_size", minBS, args.max_batch_size, log=True)
    args.dropout = trial.suggest_float("dropout", 0, 0.6,step=0.2)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    if args.first_mod.find("convnext") != -1:
        args.stochastic_depth_prob = trial.suggest_float("dropout", 0, 0.6,step=0.2)
    
    if args.master_net:
        args.kl_temp = trial.suggest_float("kl_temp", 1, 21, step=5)
        args.kl_interp = trial.suggest_float("kl_interp", 0.1, 1, step=0.1)

    value = train(args,trial)
    return value

def save_git_status(args):

    path_start = f"../models/{args.exp_id}/model{args.model_id}"

    if hasattr(args,"trial_id") and args.trial_id is not None:
        path_start += "_"+str(args.trial_id)

    cmd_list = ["git status","git diff","git rev-parse --short HEAD"]
    labels = ["status","diff","commit"]

    for labels,cmd in zip(labels,cmd_list):
        path = path_start + "_git_"+labels+".txt"
        output = subprocess.check_output(cmd.split(" "),text=True)
        with open(path,"w") as file:
            print(output,file=file)

def train(args,trial):

    if not trial is None:
        args.trial_id = trial.number

    save_git_status(args)

    if not args.only_test:
        trainLoader,_ = load_data.buildTrainLoader(args)
    else:
        trainLoader = None
    valLoader,_ = load_data.buildTestLoader(args,"val")
    
    kwargsTr = {'loader': trainLoader, 'args': args}
    kwargsVal = {'loader':valLoader,'args': args}

    # Building the net
    net = modelBuilder.netBuilder(args)

    if args.ssl:
        teach_net = modelBuilder.netBuilder(args)
        teach_net.center = torch.zeros(1, teach_net.secondModel.nbFeat).to("cuda" if args.cuda else "cpu")
        teach_net.eval()
        kwargsTr["teacher_net"] = teach_net
        kwargsVal["teacher_net"] = teach_net

    if args.regression:
        class_nb_dic = utils.make_class_nb_dic(args)
    else:
        class_nb_dic = None

    kwargsTr["class_nb_dic"] = class_nb_dic
    kwargsVal["class_nb_dic"] = class_nb_dic

    startEpoch = init_model.initialize_Net_And_EpochNumber(net, args.exp_id, args.      model_id, args.cuda, args.start_mode,args.init_path,args.optuna,ssl=args.ssl,strict=args.strict_init)

    kwargsTr["optim"],scheduler = init_model.getOptim_and_Scheduler(startEpoch,net,args)

    epoch = startEpoch
    bestEpoch, worseEpochNb = init_model.getBestEpochInd_and_WorseEpochNb(args.start_mode, args.exp_id, args.model_id, epoch)

    bestMetricVal = -np.inf
    isBetter = lambda x, y: x > y

    if args.master_net:
        kwargsTr["master_net"] = init_model.initMasterNet(args)
        kwargsVal["master_net"] = kwargsTr["master_net"]
    else:
        kwargsTr["master_net"] = None
        kwargsVal["master_net"] = None

    loss_func = SelfSuperVisedLoss() if args.ssl else SupervisedLoss(regression=args.regression,args=args)
    if args.multi_gpu:
        loss_func = torch.nn.DataParallel(loss_func)

    kwargsTr["loss_func"],kwargsVal["loss_func"] = loss_func,loss_func
    if scheduler is not None:
        print("Init lr",scheduler.get_last_lr())

    swa_net = torch.optim.swa_utils.AveragedModel(net) if args.swa else None

    if not args.only_test:

        actual_bs = args.batch_size if args.batch_size < args.max_batch_size_single_pass else args.max_batch_size_single_pass
        args.batch_per_epoch = len(trainLoader.dataset)//actual_bs if len(trainLoader.dataset) > actual_bs else 1
   
        if args.ssl:
            args,kwargsTr["optim"] = update.ssl_updates(args,kwargsTr["optim"],epoch)

        while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:
        
            kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
            kwargsTr["model"] = net
            kwargsVal["model"] = swa_net if args.swa else net

            #Training
            training_epoch(**kwargsTr)

            #Save most recent model 
            if not args.optuna:
                state_dic = net.module.state_dict() if args.swa else net.state_dict()
                torch.save(state_dic, f"../models/{args.exp_id}/model{args.model_id}_epoch{epoch}")
                previous_epoch_model = f"../models/{args.exp_id}/model{args.model_id}_epoch{epoch-1}"
                if os.path.exists(previous_epoch_model):
                    os.remove(previous_epoch_model)

            #SWA updates
            if args.swa:
                if epoch <= args.swa_start_epoch:
                    scheduler.step()
                else:
                    print("SWA update")
                    swa_net.update_parameters(net)
                    torch.optim.swa_utils.update_bn(trainLoader, swa_net)

            #Validation
            if not args.no_val:
                if epoch% args.val_freq == 0:
                    with torch.no_grad():
                        metricVal = evaluation(**kwargsVal)

                    net_to_update = swa_net if args.swa else net
                    bestEpoch, bestMetricVal, worseEpochNb = update.updateBestModel(metricVal, bestMetricVal, args.exp_id,
                                                                                args.model_id, bestEpoch, epoch,net_to_update,
                                                                                isBetter, worseEpochNb,args)
                    if trial is not None:
                        trial.report(metricVal, epoch)

            #SSL updates 
            if args.ssl:
                print("LR",scheduler.get_last_lr())
                print("Weight decay",kwargsTr["optim"].param_groups[0]["weight_decay"])
                print("Teach momentum",args.teach_momentum)
                print("Teach temp",args.teach_temp)
                args,kwargsTr["optim"] = update.ssl_updates(args,kwargsTr["optim"],epoch)
                scheduler.step()

            epoch += 1

    if trial is None:

        test_already_done = os.path.exists(f"../results/{args.exp_id}/metrics_{args.model_id}_test.csv")

        if args.run_test and ((not test_already_done) or not args.not_test_again):

            testFunc = evaluation

            kwargsTest = kwargsVal
            kwargsTest["mode"] = "test"

            testLoader,_ = load_data.buildTestLoader(args, "test")

            kwargsTest['loader'] = testLoader

            best_path = f"../models/{args.exp_id}/model{args.model_id}_best_epoch{bestEpoch}"
                
            net = init_model.preprocessAndLoadParams(best_path,args.cuda,net,ssl=args.ssl)

            kwargsTest["model"] = net
            kwargsTest["epoch"] = bestEpoch

            with torch.no_grad():
                testFunc(**kwargsTest)

            with open("../results/{}/test_done.txt".format(args.exp_id),"a") as text_file:
                print("{},{}".format(args.model_id,bestEpoch),file=text_file)

    else:

        oldPath = "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, bestEpoch)
        os.rename(oldPath, oldPath.replace("best_epoch","trial{}_best_epoch".format(trial.number)))

        with open("../results/{}/{}_{}_valRet.csv".format(args.exp_id,args.model_id,trial.number),"w") as text:
            print(metricVal,file=text)

        return metricVal

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--no_val', type=str2bool, help='To not compute the validation')
    argreader.parser.add_argument('--only_test', type=str2bool, help='To only compute the test')
    argreader.parser.add_argument('--val_freq', type=int, help='Frequency at which to run a validation.')

    argreader.parser.add_argument('--do_test_again', type=str2bool, help='Does the test evaluation even if it has already been done')

    argreader.parser.add_argument('--optuna', type=str2bool, help='To run a hyper-parameter study')
    argreader.parser.add_argument('--optuna_trial_nb', type=int, help='The number of hyper-parameter trial to run.')
    argreader.parser.add_argument('--opt_data_aug', type=str2bool, help='To optimise data augmentation hyper-parameter.')
    argreader.parser.add_argument('--opt_att_maps_nb', type=str2bool, help='To optimise the number of attention maps.')

    argreader.parser.add_argument('--max_batch_size', type=int, help='To maximum batch size to test.')

    argreader.parser.add_argument('--trial_id', type=int, help='The trial ID. Useful for grad exp during test')

    argreader.parser.add_argument('--log_gradient_norm_frequ', type=int, help='The step frequency at which to save gradient norm.')

    argreader.parser.add_argument('--save_output_during_validation', type=str2bool, help='To save model output during validation.')

    argreader.parser.add_argument('--distribution_learning', type=str2bool, help='To learn target distribution instead of vote. Works only for DL4IVF dataset.')

    argreader.parser.add_argument('--zmos', type=str2bool, help='To use ZMOS scores when training regression model')

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addSSLArgs(argreader)
    argreader = addRegressionArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)
    argreader = addSalMetrArgs(argreader)
    argreader = init_post_hoc_arg(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.debug:
        args.epochs = 1
        args.batch_size = 2
        args.val_batch_size = 2

    args.cuda = args.cuda and torch.cuda.is_available()

    if args.redirect_out:
        sys.stdout = open("python.out", 'w')

    torch.autograd.set_detect_anomaly(True)

    # The folders where the experience file will be written
    if not os.path.exists("../vis/{}".format(args.exp_id)):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not os.path.exists("../results/{}".format(args.exp_id)):
        os.makedirs("../results/{}".format(args.exp_id))
    if not os.path.exists("../models/{}".format(args.exp_id)):
        os.makedirs("../models/{}".format(args.exp_id))

    args = update.updateSeedAndNote(args)

    # Update the config args
    argreader.args = args
    # Write the arguments in a config file so the experiment can be re-run

    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id, args.model_id))
    print("Model :", args.model_id, "Experience :", args.exp_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.optuna:
        def objective(trial):
            return run(args,trial=trial)

        study = optuna.create_study(direction="maximize" if args.maximise_val_metric else "minimize",\
                                    storage="sqlite:///../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id), \
                                    study_name=args.model_id,load_if_exists=True)

        con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
        curr = con.cursor()

        failedTrials = 0
        for elem in curr.execute('SELECT trial_id,state FROM trials WHERE study_id == 1').fetchall():
            if elem[1] != "COMPLETE":
                failedTrials += 1

        trialsAlreadyDone = len(curr.execute('SELECT trial_id FROM trials WHERE study_id == 1').fetchall())

        if trialsAlreadyDone-failedTrials < args.optuna_trial_nb:

            studyDone = False
            while not studyDone:
                try:
                    print("N trials",args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
                    study.optimize(objective,n_trials=args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
                    studyDone = True
                except Exception as e:
                    if str(e).find("CUDA out of memory.") != -1:
                        gc.collect()
                        torch.cuda.empty_cache()
                        args.max_batch_size -= 5
                        print("New max batch size",args.max_batch_size)
                    else:
                        raise RuntimeError(e)

        curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
        query_res = curr.fetchall()

        query_res = list(filter(lambda x:not x[1] is None,query_res))

        trialIds = [id_value[0] for id_value in query_res]
        values = [id_value[1] for id_value in query_res]

        trialIds = trialIds[:args.optuna_trial_nb]
        values = values[:args.optuna_trial_nb]

        bestTrialId = trialIds[np.array(values).argmax()]

        curr.execute('SELECT param_name,param_value from trial_params WHERE trial_id == {}'.format(bestTrialId))
        query_res = curr.fetchall()

        args.only_test = True

        print("bestTrialId-1",bestTrialId-1)
        bestPath = glob.glob("../models/{}/model{}_trial{}_best_epoch*".format(args.exp_id,args.model_id,bestTrialId-1))[0]
        print(bestPath)

        copyfile(bestPath, bestPath.replace("_trial{}".format(bestTrialId-1),""))

        train(args,None)

    else:
        train(args,None)

if __name__ == "__main__":
    main()
