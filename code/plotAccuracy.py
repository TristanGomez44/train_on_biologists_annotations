import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def smooth(x, smoothFact,window='hanning'):
    """
    Smooth the graph of a list by averaging each element with its neighbors
    @param tab: the list to be smoothed
    @param smoothFact: the smoothing factor
    @return: the smoothed list
    """

    window_len = int(len(x) * smoothFact)

    if window_len==0:
        return x
    else:
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
        if window_len<3:
            return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')

        return y[window_len:-window_len+1]

def main():

    parser = argparse.ArgumentParser(description='Plot the accuracy across epoch')
    parser.add_argument('--csv_list', metavar='CSV',nargs='+',type=str,help='path to the csv files')
    parser.add_argument('--outfile', metavar='OUT',type=str,help='path to the output image file')
    parser.add_argument('--yaxis', metavar='YAXIS',type=str,help='Name of the y-axis')
    parser.add_argument('--xaxis', metavar='XAXIS',type=str,help='Name of the x-axis',default="Epochs")
    parser.add_argument('--title', metavar='TITLE',type=str,help='Title of the plot')
    parser.add_argument('--smooth', metavar='SMOOTH',type=float,default=0.1,help='The proportion of neighboring point \
                        to take into account during the smoothing of the function')
    parser.add_argument('--label_list', metavar='LABELS',nargs='+',type=str,help='The list of labels to give to each curve')
    parser.add_argument('--color_list', metavar='COLORS',nargs='+',type=str,help='The list of colors to give to each curve')
    parser.add_argument('--linestyle_list', metavar='LINESTYLE',nargs='*',type=str,help='The list of line style to give to each curve')
    parser.add_argument('--leg_nb', metavar='LEGNB',type=int,help='The number of legend to plot')
    parser.add_argument('--leg_names', metavar='LEGNAMES',nargs='+',type=str,help='The name of each legend ')
    parser.add_argument('--plot_optim', metavar='OPTIM',nargs='+',type=str,help='For each curve, whether to plot the optimum or not')
    parser.add_argument('--up_ylim', metavar='UPYLIM',type=float,default=1,help='The upper limit for the y axis ')
    parser.add_argument('--bottom_ylim', metavar='BOTYLIM',type=float,default=0,help='The lower limit for the y axis')
    parser.add_argument('--max', action='store_true',help='To plot the maximum of each curve with a circle. \
                        Otherwise, it\'s the minimum which will be emphasized')
    parser.add_argument('--vbar', action='store_true',help='To plot vertical var on the optimums')
    parser.add_argument('--epochs', metavar='NB',type=int,help='The maximum number of epoch to plot',default=100)

    args = parser.parse_args()

    loc_list = ['upper right','lower right','center right','upper right']
    markerList = [".","*","s"]

    modelNbPerLegend = int(len(args.csv_list)/args.leg_nb)
    modelCounter = 0
    legendCounter = 0

    handles = [[] for i in range(args.leg_nb)]
    handlesLabel = [[] for i in range(args.leg_nb)]

    plot = plt.figure(figsize=(9,5))
    ax1 = plot.add_subplot(111)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width*0.8 , box.height])

    for i in range(len(args.csv_list)):

        csv = np.genfromtxt(args.csv_list[i],delimiter=',')[1:,1:]
        csv = np.array(sorted(csv,key=lambda x:x[0]))
        csv[:,1] = smooth(csv[:,1],args.smooth)

        handlesLabel[legendCounter].append(args.label_list[i])
        handles[legendCounter].append(ax1.plot(csv[:args.epochs,0],csv[:args.epochs,1],color=args.color_list[i],linestyle=args.linestyle_list[i])[0])

        if args.plot_optim and args.plot_optim[i] == "True":
            print("plotting min")
            if args.max:
                arg = np.argmax(csv[:,1])
                handlesLabel[legendCounter].append("Maximum of {}".format(args.label_list[i]))
            else:
                arg = np.argmin(csv[:,1])
                handlesLabel[legendCounter].append("Minimum of {}".format(args.label_list[i]))

            x_max,ymax = csv[arg,0],csv[arg,1]

            handles[legendCounter].append(ax1.plot(x_max,ymax,markerList[modelCounter],color=args.color_list[i],markersize=10)[0])

            if args.vbar:
                plt.axvline(x=x_max,color=args.color_list[i],linestyle='-.',linewidth=0.5)
                plt.text(x_max, (2/(0+1))*1.5*args.up_ylim/100, " {} \n Step {}".format(args.leg_names[legendCounter],int(x_max)), fontsize=10)

        if modelCounter != modelNbPerLegend -1 :
            #Keeping the same legend
            modelCounter += 1
        else:
            #Switching to another legend
            modelCounter=0
            legendCounter  += 1

    plt.xlabel(args.xaxis)
    plt.ylabel(args.yaxis)
    plt.ylim(args.bottom_ylim,args.up_ylim)
    plt.title(args.title)
    plt.grid(True)

    for i in range(args.leg_nb):
        leg = plot.legend(handles[i],handlesLabel[i],title=args.leg_names[i],bbox_to_anchor=(1, i*0.2+0.5))
        plot.gca().add_artist(leg)

    plt.savefig(args.outfile)

if __name__ == "__main__":
    main()
