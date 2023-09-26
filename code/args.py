import sys
import argparse
import configparser

def str2bool(v):
    '''Convert a string to a boolean value'''
    if v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2FloatList(x):

    '''Convert a formated string to a list of float value'''
    if len(x.split(",")) == 1:
        return float(x)
    else:
        return [float(elem) for elem in x.split(",")]
def strToStrList(x):
    if x == "None":
        return []
    else:
        return x.split(",")

def str2StrList(x):
    '''Convert a string to a list of string value'''
    return x.split(" ")

class ArgReader():
    """
    This class build a namespace by reading arguments in both a config file
    and the command line.

    If an argument exists in both, the value in the command line overwrites
    the value in the config file

    This class mainly comes from :
    https://stackoverflow.com/questions/3609852/which-is-the-best-way-to-allow-configuration-options-be-overridden-at-the-comman
    Consulted the 18/11/2018

    """

    def __init__(self,argv):
        ''' Defines the arguments used in several scripts of this project.
        It reads them from a config file
        and also add the arguments found in command line.

        If an argument exists in both, the value in the command line overwrites
        the value in the config file
        '''

        # Do argv default this way, as doing it in the functional
        # declaration sets it at compile time.
        if argv is None:
            argv = sys.argv

        # Parse any conf_file specification
        # We make this parser with add_help=False so that
        # it doesn't parse -h and print help.
        conf_parser = argparse.ArgumentParser(
            description=__doc__, # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False
            )
        conf_parser.add_argument("-c", "--conf_file",
                            help="Specify config file", metavar="FILE")
        args, self.remaining_argv = conf_parser.parse_known_args()

        defaults = {}

        if args.conf_file:
            config = configparser.ConfigParser()
            config.read([args.conf_file])
            defaults.update(dict(config.items("default")))

        # Parse rest of arguments
        # Don't suppress add_help here so it will handle -h
        self.parser = argparse.ArgumentParser(
            # Inherit options from config_parser
            parents=[conf_parser]
            )
        self.parser.set_defaults(**defaults)

        self.parser.add_argument('--log_interval', type=int, metavar='M',
                            help='The number of batchs to wait between each console log')
        self.parser.add_argument('--num_workers', type=int,metavar='NUMWORKERS',
                            help='the number of processes to load the data. num_workers equal 0 means that it’s \
                            the main process that will do the data loading when needed, num_workers equal 1 is\
                            the same as any n, but you’ll only have a single worker, so it might be slow')
        self.parser.add_argument('--cuda', type=str2bool, metavar='S',
                            help='To run computations on the gpu')
        self.parser.add_argument('--multi_gpu', type=str2bool, metavar='S',
                            help='If cuda is true, run the computation with multiple gpu')
        self.parser.add_argument('--debug', type=str2bool,metavar='BOOL',
                            help="To run only a few batch of training and a few batch of validation")
        self.parser.add_argument('--benchmark', type=str2bool,metavar='BOOL',
                            help="To check hardware occupation during training. Results will be put in the ../results/<exp_id> folder.")

        self.parser.add_argument('--redirect_out', type=str2bool,metavar='BIDIR',
                            help='If true, the standard output will be redirected to a file python.out')
        self.parser.add_argument('--note', type=str,metavar='NOTE',
                            help="A note on the model")

        self.parser.add_argument('--epochs', type=int, metavar='N',
                            help='maximum number of epochs to train')

        #Arg to share
        self.parser.add_argument('--model_id', type=str, metavar='IND_ID',
                            help='the id of the individual model')
        self.parser.add_argument('--exp_id', type=str, metavar='EXP_ID',
                            help='the id of the experience')
        self.parser.add_argument('--seed', type=int, metavar='S',help='Seed used to initialise the random number generator.')


        self.args = None

    def getRemainingArgs(self):
        ''' Reads the comand line arg'''

        self.args = self.parser.parse_args(self.remaining_argv)

    def writeConfigFile(self,filePath):
        """ Writes a config file containing all the arguments and their values"""

        config = configparser.ConfigParser()
        config.add_section('default')

        for k, v in  vars(self.args).items():
            config.set('default', k, str(v))

        with open(filePath, 'w') as f:
            config.write(f)

def addInitArgs(argreader):
    argreader.parser.add_argument('--start_mode', type=str, metavar='SM',
                                  help='The mode to use to initialise the model. Can be \'scratch\' or \'fine_tune\'.')
    argreader.parser.add_argument('--init_path', type=str, metavar='SM',
                                  help='The path to the weight file to use to initialise the network')
    argreader.parser.add_argument('--strict_init', type=str2bool, metavar='SM',
                                  help='Set to True to make torch.load_state_dict throw an error when not all keys match (to use with --init_path)')

    return argreader

def addValArgs(argreader):

    argreader.parser.add_argument('--metric_early_stop', type=str, metavar='METR',
                                  help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_val_metric', type=str2bool, metavar='BOOL',
                                  help='If true, The chosen metric for chosing the best model will be maximised')
    argreader.parser.add_argument('--max_worse_epoch_nb', type=int, metavar='NB',
                                  help='The number of epochs to wait if the validation performance does not improve.')
    argreader.parser.add_argument('--run_test', type=str2bool, metavar='NB',
                                  help='Evaluate the model on the test set')
    argreader.parser.add_argument('--not_test_again',action="store_true",
                                  help='To not re-evaluate if test is already done')
    return argreader

def init_post_hoc_arg(argreader):
    argreader.parser.add_argument('--att_metrics_post_hoc', type=str, help='The post-hoc method to use instead of the model ')
    argreader.parser.add_argument('--img_nb_per_class', type=int, help='The nb of images on which to compute the att metric.')    
    argreader.parser.add_argument('--ablationcam_batchsize', type=int,default=2048)
    return argreader


def addLossTermArgs(argreader):
    argreader.parser.add_argument('--nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function.')
    argreader.parser.add_argument('--nce_weight', type=float, metavar='FLOAT',
                                  help='The weight of the saliency metric mask in the loss function. Can be set to "scheduler".')
    
    argreader.parser.add_argument('--nce_weight_sched', type=str2bool, metavar='FLOAT',
                                  help='Whether or not to use a scheduler for nce weight.')
    
    argreader.parser.add_argument('--nce_sched_start', type=float, metavar='FLOAT',
                                  help='The initial value of nce_weight loss term.')
    argreader.parser.add_argument('--nce_norm', type=str2bool, metavar='FLOAT',
                                  help='To add the NCE normalisation (i.e. cross entropy and negatives terms)')
    argreader.parser.add_argument('--focal_weight', type=float, metavar='FLOAT',
                                  help='The weight of the focal loss term.')
    argreader.parser.add_argument('--adv_weight', type=float, metavar='FLOAT',
                                  help='The weight of the adversarial loss term to ensure masked representations are indistinguishable from regular representations.')
    argreader.parser.add_argument('--map_sim_term_weight', type=float, metavar='FLOAT',
                                  help='Weight of the map similarity term.')
    argreader.parser.add_argument('--task_to_train', type=str, metavar='FLOAT',
                                  help='The task to train. Set to "all" to train all tasks. Task can take value among [ICM,TE,EXP].')

    return argreader

def addSalMetrArgs(argreader):
    argreader.parser.add_argument('--sal_metr_otherimg',type=str2bool, help='To fill removed image areas with parts of another image.')
    argreader.parser.add_argument('--sal_metr_bckgr',type=str, help='The filling method to use for saliency metrics. Ignored if --sal_metr_otherimg is True.')
    argreader.parser.add_argument('--sal_metr_non_cum',type=str2bool, help='To not accumulate pertubations when computing saliency metrics.')
    argreader.parser.add_argument('--sal_metr_mask', type=str2bool, help='To apply the masking of attention metrics during training.')
    argreader.parser.add_argument('--sal_metr_mask_prob', type=float, help='The probability to apply saliency metrics masking.')
    argreader.parser.add_argument('--sal_metr_mask_remove_masked_obj',type=str2bool, help='Set to True to remove terms masked by the DAUC and ADD metrics.')
    return argreader