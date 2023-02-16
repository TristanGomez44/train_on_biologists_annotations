import glob,os,configparser
import torch
import utils

def getOptim_and_Scheduler(optimStr, lr,momentum,weightDecay,useScheduler,lastEpoch,net,step_size=2,gamma=0.9):

    if optimStr != "AMSGrad":
        optimConst = getattr(torch.optim, optimStr)
        if optimStr == "SGD":
            kwargs = {'lr':lr,'momentum': momentum,"weight_decay":weightDecay}
        elif optimStr == "Adam":
            kwargs = {'lr':lr,"weight_decay":weightDecay}
        elif optimStr == "AdamW":
            kwargs = {'lr':lr,"weight_decay":weightDecay}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(optimStr))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'lr':lr,'amsgrad': True,"weight_decay":weightDecay}

    optim = optimConst(net.parameters(), **kwargs)

    if useScheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
        for _ in range(lastEpoch-1):
            scheduler.step()
        
        print("StepLR:",step_size,gamma,scheduler.get_last_lr())
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

def preprocessAndLoadParams(init_path,cuda,net):
    print("Init from",init_path)
    params = torch.load(init_path, map_location="cpu" if not cuda else None)
    params = addOrRemoveModule(params,net)
    res = net.load_state_dict(params, False)

    # Depending on the pytorch version the load_state_dict() method can return the list of missing and unexpected parameters keys or nothing
    if not res is None:
        missingKeys, unexpectedKeys = res
        if len(missingKeys) > 0:
            print("missing keys")
            for key in missingKeys:
                print(key)
        if len(unexpectedKeys) > 0:
            print("unexpected keys")
            for key in unexpectedKeys:
                print(key)

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
