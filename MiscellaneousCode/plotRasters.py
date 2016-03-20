
def raster(event_times_list, color='k'):
    """
        Creates a raster plot
        Parameters
        ----------
        event_times_list : iterable
        a list of event time iterables
        color : string
        color of vlines
        Returns
        -------
        ax : an axis containing the raster plot
        
        """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    ax.set_xticks(numpy.arange(0,3600,120))
    xax = ax.get_xaxis()
    xax.set_tick_params(direction='in',color='r',length=10,width=1)
    ax.axes.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.yaxis.set_ticks([])
    #for label in ax.xaxis.get_ticklabels()[1:-1]:
    #    label.set_visible(False)
    return ax

def raster2(event_times_list, color='k'):
    """
        Creates a raster plot
        Parameters
        ----------
        event_times_list : iterable
        a list of event time iterables
        color : string
        color of vlines
        Returns
        -------
        ax : an axis containing the raster plot
        """
    ax = plt.gca()
    for ith, trial in enumerate(event_times_list):
        plt.vlines(trial, ith + .5, ith + 1.5, color=color)
    plt.ylim(.5, len(event_times_list) + .5)
    ax.set_xticks(numpy.arange(0,3600,120))
    xax = ax.get_xaxis()
    xax.set_tick_params(direction='in',color='r',length=10,width=1)
    ax.axes.xaxis.set_ticklabels(numpy.arange(0,30,1))
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.yaxis.set_ticks([])
    #for label in ax.xaxis.get_ticklabels()[1:-1]:
    #    label.set_visible(False)
    return ax