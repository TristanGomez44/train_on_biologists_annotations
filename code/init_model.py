import glob,os,configparser
import torch
import utils

def getOptim_and_Scheduler(lastEpoch,net,args):

    if args.optim != "AMSGrad":
        optimConst = getattr(torch.optim, args.optim)
        if args.optim == "SGD":
            kwargs = {'lr':args.lr,'momentum': args.momentum,"weight_decay":args.weight_decay}
        elif args.optim == "Adam":
            kwargs = {'lr':args.lr,"weight_decay":args.weight_decay}
        elif args.optim == "AdamW":
            kwargs = {'lr':args.lr,"weight_decay":args.weight_decay}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'lr':args.lr,'amsgrad': True,"weight_decay":args.weight_decay}

    optim = optimConst(net.parameters(), **kwargs)

    if args.swa:
        def warmup_lambda_func(epoch):
            alpha = epoch/args.warmup_epochs
            lr = (alpha*args.lr+(1-alpha)*args.warmup_lr)/args.lr
            return lr
    
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lambda_func)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.swa_start_epoch-args.warmup_epochs,args.swa_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optim, [warmup_scheduler,cosine_scheduler],milestones=[args.warmup_epochs])

        for _ in range(lastEpoch-1):
            scheduler.step()
    else:
        scheduler = None

    return optim, scheduler

def initialize_Net_And_EpochNumber(net, exp_id, model_id, cuda, start_mode, init_path,optuna):

    if start_mode == "auto":
        if (not optuna) and len(glob.glob("../models/{}/model{}_epoch*".format(exp_id, model_id))) > 0:
            start_mode = "fine_tune"
        else:
            start_mode = "scratch"
        print("Autodetected mode", start_mode)

    if start_mode == "scratch":

        # Saving initial parameters
        torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id, model_id))
        startEpoch = 1

    elif start_mode == "fine_tune":

        if init_path == "None":
            init_path = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id, model_id)), key=utils.findLastNumbers)[-1]

        net = preprocessAndLoadParams(init_path,cuda,net)

        filename = os.path.basename(init_path)
        model_id_init_path = filename.split("model")[1]
        split_keyword = "_best" if filename.find("best") != -1 else "_epoch"      
        model_id_init_path = model_id_init_path.split(split_keyword)[0]
            
        if model_id_init_path == model_id:
            startEpoch = utils.findLastNumbers(init_path)+1
        else:
            startEpoch = 1

    return startEpoch

def removeExcessModule(params):
    new_params = {}
    for key in params:
        new_key = key.replace("module.module.","module.")
        new_params[new_key] = params[key]
    return new_params

def preprocessAndLoadParams(init_path,cuda,net,verbose=True):
    if verbose:
        print("Init from",init_path)
    params = torch.load(init_path, map_location="cpu" if not cuda else None)
    params = addOrRemoveModule(params,net)
    params = removeExcessModule(params)
    res = net.load_state_dict(params, False)

    missingKeys, unexpectedKeys = res
    if len(missingKeys) > 0:
        print("missing keys")
        for key in missingKeys:
            print(key)
    if len(unexpectedKeys) > 0:
        print("unexpected keys")
        for key in unexpectedKeys:
            print(key)
        if len(unexpectedKeys)==1 and "n_averaged" in unexpectedKeys[0]:
            unexpectedKeys = []

    assert len(missingKeys) == 0 and len(unexpectedKeys)==0,"Some keys were missing/unexpected. See message above."
        
    return net

def addOrRemoveModule(params,net):
    # Checking if the key of the model start with "module."
    startsWithModule = (list(net.state_dict().keys())[0].find("module.") == 0)

    if startsWithModule:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = "module." + key if key.find("module") == -1 else key
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    else:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = key.split('.')
            if keyFormat[0] == 'module':
                keyFormat = '.'.join(keyFormat[1:])
            else:
                keyFormat = '.'.join(keyFormat)
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    return params

def getBestEpochInd_and_WorseEpochNb(start_mode, exp_id, model_id, epoch):
    if start_mode == "scratch":
        bestEpoch = epoch
        worseEpochNb = 0
    else:
        bestModelPaths = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id, model_id))
        if len(bestModelPaths) == 0:
            bestEpoch = epoch
            worseEpochNb = 0
        elif len(bestModelPaths) == 1:
            bestModelPath = bestModelPaths[0]
            bestEpoch = int(os.path.basename(bestModelPath).split("epoch")[1])
            worseEpochNb = epoch - bestEpoch
        else:
            raise ValueError("Wrong number of best model weight file : ", len(bestModelPaths))

    return bestEpoch, worseEpochNb

def initMasterNet(args):
    config = configparser.ConfigParser()

    config.read("../models/{}/{}.ini".format(args.exp_id,args.m_model_id))
    args_master = utils.Bunch(config["default"])

    args_master.multi_gpu = args.multi_gpu

    argDic = args.__dict__
    mastDic = args_master.__dict__

    for arg in mastDic:
        if arg in argDic:
            if not argDic[arg] is None:
                if not type(argDic[arg]) is bool:
                    if mastDic[arg] != "None":
                        mastDic[arg] = type(argDic[arg])(mastDic[arg])
                    else:
                        mastDic[arg] = None
                else:
                    if arg != "multi_gpu" and arg != "distributed":
                        mastDic[arg] = str2bool(mastDic[arg]) if mastDic[arg] != "None" else False
            else:
                mastDic[arg] = None

    for arg in argDic:
        if not arg in mastDic:
            mastDic[arg] = argDic[arg]

    master_net = modelBuilder.netBuilder(args_master)

    best_paths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.m_model_id))

    if len(best_paths) > 1:
        raise ValueError("Too many best path for master")
    if len(best_paths) == 0:
        print("Missing best path for master")
    else:
        bestPath = best_paths[0]
        params = torch.load(bestPath, map_location="cpu" if not args.cuda else None)

        for key in params:
            if key.find("firstModel.attention.1.weight") != -1:

                if params[key].shape[0] < master_net.state_dict()[key].shape[0]:
                    padd = torch.zeros(1,params[key].size(1),params[key].size(2),params[key].size(3)).to(params[key].device)
                    params[key] = torch.cat((params[key],padd),dim=0)
                elif params[key].shape[0] > master_net.state_dict()[key].shape[0]:
                    params[key] = params[key][:master_net.state_dict()[key].shape[0]]

        params = addOrRemoveModule(params,master_net)
        master_net.load_state_dict(params, strict=True)

    master_net.eval()

    return master_net
