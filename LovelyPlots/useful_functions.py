import numpy as np
import pandas as pd
import statsmodels.api as sm
idx = pd.IndexSlice
from matplotlib import pyplot as plt
import seaborn as sns
from cycler import cycler
from datetime import datetime
import matplotlib as mpl
import os
import re
import shutil
import zipfile
from string import printable as character_list
import xmltodict
from scipy import stats
import cProfile
import io
import pstats
from linearmodels.panel import compare


def get_sheet_details(file_path):
    sheets = []
    file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
    # Make a temporary directory with the file name
    directory_to_extract_to = os.path.join(os.getcwd(), file_name)
#     print(file_path)
    try:
        os.mkdir(directory_to_extract_to)
    except FileExistsError:
        shutil.rmtree(directory_to_extract_to)
        os.mkdir(directory_to_extract_to)


    # Extract the xlsx file as it is just a zip file
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

    # Open the workbook.xml which is very light and only has meta data, get sheets from it
    path_to_workbook = os.path.join(directory_to_extract_to, 'xl', 'workbook.xml')
    with open(path_to_workbook, 'r') as f:
        xml = f.read()
        dictionary = xmltodict.parse(xml)
        for sheet in dictionary['workbook']['sheets']['sheet']:
            sheet_details = {
                'id': sheet['@sheetId'], # can be @sheetId for some versions
                'name': sheet['@name'] # can be @name
            }
            sheets.append(sheet_details)

    # Delete the extracted files directory
    shutil.rmtree(directory_to_extract_to)
    return sheets

def get_sheet_details(file_path):
    sheet_names = []
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        xml = zip_ref.open(r'xl/workbook.xml').read()
        dictionary = xmltodict.parse(xml)

        if not isinstance(dictionary['workbook']['sheets']['sheet'], list):
            sheet_names.append(dictionary['workbook']['sheets']['sheet']['@name'])
        else:
            for sheet in dictionary['workbook']['sheets']['sheet']:
                sheet_names.append(sheet['@name'])
    return sheet_names

def can_be_int(i):
    try:
        int(i)
        return True
    except:
        return False
    
def init_plot_simple():
    mpl.rcParams['legend.framealpha'] = 1
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.axis'] = 'both'
    mpl.rcParams['grid.color'] = '0.9'
    mpl.rcParams['grid.linewidth'] = 1
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['axes.titleweight'] = 'bold'
    sns.set_palette('Dark2')

def init_plot2(fontsize=20,figsize=(8,5.5),font='Arial',font_family='sans-serif',linewidth=4,font_style='bold',
               have_axes=True,dpi=50,marker='None',markersize=12,markeredgewidth=1.0,markeredgecolor=None,markerfacecolor=None, 
               markercycler=False, linestylecycler=False, cmap='Dark2', n_colors=8, **kwargs):
    '''Sets default plot formats.
    Potential inputs: fontsize, figsize, font,
    font_family, font_style, linewidth, have_axes,
    dpi, marker, markersize, markeredgewidth,
    markeredgecolor, markerfacecolor.
    have_axes: determines whether there is a border
    on the plot. Also has **kwargs so that any other
    arguments that can be passed to mpl.rcParams.update
    that were not listed above.

    markercycler: bool, if True, sets the cycler to include a set of markers
    linestylecycler: bool, if True, sets the cycler to include the 4 linestyles

    cmap can take any matplotlib colormap string.'''
    import matplotlib as mpl
    params = {
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'axes.titlesize':fontsize+1,
        'axes.titleweight':font_style,
        'figure.titlesize':fontsize+1,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.titlesize': fontsize+2,
        'figure.titlesize': fontsize+2,
        'text.usetex': False,
        'figure.figsize': figsize,
        'lines.linewidth': linewidth,
        'lines.solid_capstyle': 'round',
        'legend.framealpha': 1,
        'legend.frameon': False,
        'mathtext.default': 'regular',
        'axes.linewidth': 2/3*linewidth,
        'xtick.direction': 'in', # in, out, inout
        'ytick.direction': 'in', # in, out, inout
        'xtick.major.size': 7,
        'xtick.major.width': 2,
        'xtick.major.pad': 3.5,
        'ytick.major.size': 7,
        'ytick.major.width': 2,
        'ytick.major.pad': 3.5,
        'font.family': font_family,
        'font.'+font_family: font,
        'figure.dpi': dpi,
        'lines.marker': marker,
        'lines.markersize':markersize,
        'lines.markeredgewidth':markeredgewidth,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.transparent': True,
        'savefig.bbox': 'tight',
        'axes.facecolor': 'white',
        'axes.edgecolor':'k',
        }

    mpl.rcParams.update(params)
    mpl.rcParams['axes.spines.left'] = have_axes
    mpl.rcParams['axes.spines.right'] = have_axes
    mpl.rcParams['axes.spines.top'] = have_axes
    mpl.rcParams['axes.spines.bottom'] = have_axes
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.axis'] = 'both'
    mpl.rcParams['grid.color'] = '0.9'
    mpl.rcParams['grid.linewidth'] = 1
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['axes.labelweight'] = font_style
    mpl.rcParams['font.weight'] = font_style

    sns.set_palette(cmap)

    if markeredgecolor != None:
        mpl.rcParams['lines.markeredgecolor'] = markeredgecolor
    if markerfacecolor != None:
        mpl.rcParams['lines.markerfacecolor'] = markerfacecolor
    try:
        cmap_colors = plt.get_cmap(cmap)
    except:
        cmap_colors = mpl.cm.get_cmap(cmap)
    colors = [cmap_colors(i) for i in np.linspace(0,1,n_colors)]
    
    default_cycler = cycler('color',colors)
    if markercycler:
        markers = ["o","s","^","v","p","8","P","X"]
        if n_colors>8: markers = np.tile(markers,int(np.floor(n_colors/8)+1))
        markers = markers[:n_colors]
        default_cycler = default_cycler+cycler('marker',markers)
    if linestylecycler:
        linestyles = ['-','--',':','-.']
        if n_colors>4: linestyles = np.tile(linestyles,int(np.floor(n_colors/4)+1))
        linestyles = linestyles[:n_colors]
        default_cycler = default_cycler+cycler('linestyle',linestyles)
    mpl.rcParams['axes.prop_cycle'] = default_cycler
    mpl.rcParams.update(**kwargs)

def plot_back_to_default_style():
    mpl.rcParams.update(mpl.rcParamsDefault)
    
def custom_legend(ax, labels, shapes, colors, kwargs=None, legend_kwargs={}):
    """
    ax:     axes on which to plot the legend
    labels: list of strings for labels of each element
    shapes: list of strings for type of shape for each element, can be `line` or `patch`
    colors: list of strings for colors of each element
    kwargs: list of dicts with additional arguments to be passed to each line or patch function
    legend_kwargs: dict, arguments to be passed to the legend function call

    all among labels, shapes, colors, and kwargs must be the same length (kwargs can also be left as None)
    """
    line = mpl.lines.Line2D
    patch = mpl.patches.Patch
    kwargs = [{} for i in range(len(labels))] if kwargs is None else kwargs
    if type(shapes)==str:
        shapes = np.repeat([shapes],len(labels))
    legend_elements = [
        line([0], [0], color=colors[i], label=labels[i], **kwargs[i]) if shapes[i]=='line' else 
        patch(facecolor=colors[i], label=labels[i], **kwargs[i]) for i in range(len(labels))
    ]
    ax.legend(handles=legend_elements, **legend_kwargs)

def close_all_plots():
    for _ in range(100):
        plt.close()

def is_pareto_efficient_dumb(costs):
    """
    From https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient
        
def reduce_mem_usage(df,inplace=False):
    '''Returns dataframe with columns changed to have dtypes of minimum
    size for the values contained within. Does not adjust object dtypes.
    From https://www.kaggle.com/gemartin/load-data-reduce-memory-usage'''
    if inplace:
        props = df
    else:
        props = df.copy()
    props = props.drop_duplicates().T.drop_duplicates().T
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    progress = int(np.floor(len(props.columns)/10))
    n = 1
    for num, col in enumerate(props.columns):
        prop_col = props[col]
        try:
            col_type = prop_col.dtypes.values[0]
        except Exception as e:
#             print(e)
            if 'numpy.dtype[' in str(e):
                col_type = prop_col.dtypes


        if col_type != object:  # Exclude strings

            # Print current column type
#             print("******************************")
#             print("Column: ",col)
#             print("dtype before: ",col_type)

            # make variables for Int, max and min
            IsInt = False
            try:
                mx = prop_col.max().max()
                mn = prop_col.min().min()
            except:
                mx = prop_col.max()
                mn = prop_col.min()


            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(prop_col).all().all():
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)

            # test if column can be converted to an integer
            asint = prop_col.fillna(0).astype(np.int64)
            result = (prop_col - asint)
            result = result.sum().sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            try:
                col_type = props[col].dtypes.values[0]
            except Exception as e:
#                 print(e)
                if 'numpy.dtype[float64]' in str(e):
                    col_type = props[col].dtypes
#             print("dtype after: ",col_type)
#             print("******************************")
        if num == progress*n:
            print('{}0% complete'.format(n))
            n+=1

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

def twinx2(ax1,tw,n=2):
    '''Primary axis, secondary axis, number of digits to round to (default 2), does not return anything.
    Sets up secondary y-axis to be aligned with the primary axis ticks.'''
    l = ax1.get_ylim()
    l2 = tw.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax1.get_yticks())
    tw.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    tw.grid(None)
    tw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.'+str(n)+'f'))

def do_a_regress(x,y,ax=0,intercept=True,scatter_color='tab:blue',line_color='k',
                 xlabel='independent var',ylabel='dependent var',log=False,print_log=False,
                 plot=True,loc='upper left',add_labels_bool=False,force_predict_v_actual=False):
    '''Performs regression between two pandas series or dataframes.
    x: pd.Series() or pd.DataFrame() of the independent variable(s)
    y: pd.Series() or pd.DataFrame() of the dependent variable
    ax: matplotlib axes to plot on, defaults to creating new figure
    intercept: boolean, whether or not to give the regression a
      constant, default True
    scatter_color: matplotlib interpretable color, default tab:blue
    line_color: matplotlib interpretable color, default black (k)
    xlabel: string, overwrite series or column name for independent
      variable
    ylabel: string, overwrite series or column name for dependent
      variable
    plot: boolean, whether or not to plot the regression result
    log: bool, whether to log transform both x and y
    print_log: bool, whether to print x/y values lost during log
    loc: text location of regression equation if plotting. Options:
      bottom right, lower right, upper left (default), or upper right
    add_labels: boolean, adds index values to each scatterplot point,
      not recommended for large datasets, default False
    force_predict_v_actual: boolean, whether to force the plot to
      plot the regression predicted values vs the actual values
      instead of y vs x. Default False but force True if there is
      more than one independent variable given.
    Returns tuple of (series with model parameters, fitted model) if
      force_predict_v_actual==False.
    If force_predict_v_actual is True, returns tuple of
      (series with model parameters, fitted predicted vs actual model,
      fitted y vs x model)'''
    if type(x) == pd.core.frame.DataFrame:
        if x.shape[1]>1:
            force_predict_v_actual = True
        elif x.shape[1]==1:
            x = x[x.columns[0]]
    try:
        x.name == None
    except:
        x.name = xlabel
    try:
        y.name == None
    except:
        y.name = ylabel
    if x.name == None:
        x.name = xlabel
    if xlabel != 'independent var':
        x.name = xlabel
    if ylabel != 'dependent var':
        y.name = ylabel
    if log:
        if len(x[x>0]) != len(x) and print_log:
            print(f'{len(x[x>0])} negative/zero/nan x values lost')
        x = x[x>0]
        if len(y[y>0]) != len(y) and print_log:
            print(f'{len(y[y>0])} negative/zero/nan y values lost')
        y = y[y>0]
        ind = np.intersect1d(x.index,y.index)
        if (len(ind) != len(x.index) or len(ind) != len(y.index)) and print_log:
            print(f'{len(x.index)-len(ind)} x values lost, {len(y.index)-len(ind)} y values lost to unaligned indices')
        x,y = np.log(x.loc[ind]), np.log(y.loc[ind])
        x.name, y.name = 'log('+str(x.name)+')', 'log('+str(y.name)+')'

    if intercept:
        x_i = sm.add_constant(x)
    else:
        x_i = x.copy()
    m = sm.GLS(y,x_i,missing='drop').fit(cov_type='HC3')

    if plot and not force_predict_v_actual:
        if type(ax) == int:
            fig,ax = plt.subplots()
        ax.scatter(x,y,color=scatter_color)
        if add_labels_bool:
            add_labels(x,y,ax)
        try:
            x = x.loc[[x.idxmax(),x.idxmin()]]
        except:
            x = x.loc[[x.idxmax()[0],x.idxmin()[0]]]
        if intercept:
            ax.plot(x, m.params['const'] + m.params[x.name]*x,label='Best-fit line',color=line_color)
        else:
            ax.plot(x, m.params[x.name]*x,label='Best-fit line',color=line_color)

        add_regression_text(m,x,y,loc=loc,ax=ax)

        ax.set(xlabel=x.name, ylabel=y.name)

    elif force_predict_v_actual:
        y_predicted = m.predict(x_i)
        if plot:
            if type(ax)==int:
                fig,ax = plt.subplots()
            y.name = 'Actual'
            y_predicted.name = 'Predicted'
            m_predict_v_actual = do_a_regress(y,y_predicted,ax=ax,intercept=intercept,
                                              scatter_color=scatter_color,line_color=line_color,
                                              xlabel='Actual',ylabel='Predicted',plot=plot,loc=loc,
                                              add_labels_bool=add_labels_bool,force_predict_v_actual=False)[1]


    if force_predict_v_actual:
        return pd.Series(m.params).rename({x.name:'slope','x1':'slope'}),m_predict_v_actual,m
    else:
        return pd.Series(m.params).rename({x.name:'slope','x1':'slope'}),m

def easy_subplots(nplots, ncol=None, height_scale=1,width_scale=1,use_subplots=False,width_ratios=None,height_ratios=None,figsize=None,**kwargs):
    '''sets up plt.subplots with the correct number of rows and columns and figsize,
    given number of plots (nplots) and the number of columns (ncol).
    Option to make figures taller/shorter or wider/narrower by changing height_scale
    or width_scale.

    Can also give additional arguments to either plt.figure or plt.subplots using **kwargs.
    ------------
    Inputs:
    - nplots: list/array/int, number of plots you want to use, can also be the
      list of variables you want to iterate over (automatically takes the len
      of any non-int input)
    - ncol: int or None. Sets number of columns in the plots. If None, will try
      set to 3 if nplots is a multiple of 3. If nplots<4, will use nplots.
      Otherwise, uses 4.
    - height_scale: amount to change the default height, default is 1. Overwritten
      if figsize is used.
    - width_scale: amount to change the default width, default is 1. Overwritten
      if figsize is used.
    - use_subplots: True to use plt.subplots to create axes, False to use
      plt.figure, which then allows dpi to be specified (can always change dpi
      after the fact by using fig.set_dpi(50). use_subplots is always set to
      True when any value is given for height_ratios or width_ratios.
    - width_ratios: None or list/array. Allows changing the relative widths of
      each column of plots. Default is np.repeat(1,ncol), can give a list of
      length ncol.
    - height_ratios: None or list/tuple/array. Allows changing the relative heights of
      each row of plots. Default is np.repeat(1,nrows), can give a list of
      length nrows (nrows=int(np.ceil(nplots/ncol)).
    - figsize: None or list/tuple/array. If given, overwrites height_scale and width_scale,
      and is used the same way as in plt.subplots() or plt.figure()
    - **kwargs: dictionary? Still don`t fully understand how this works but it works
      as a dictionary. Give a dictionary to pass additional values to plt.figure()
      or plt.subplots(), where keys are the variables you want to set.

    ------------
    Additional tip: if you want more subplot size functionality than this approach
    allows, use plt.subplot_mosaic(). Most of the inputs are the same as for
    plt.subplots() but you can also give the input mosaic=
    `AAE
     C.E`
    Which would give a top row where plot A takes the first two columns, plot E
    spans two rows on the right side, and plot C is one row/column tall/wide in
    the lower left. Could add this functionality to this function at some point.
    plt.subplot_mosaic returns a dictionary of the subplot names you give.
    ------------
    Returns:
          fig,ax (ax is a flattened array of the axes in fig)
    '''
    if type(nplots) != int:
        nplots = len(nplots)
    if nplots <= 4 and ncol is None:
        ncol = nplots
    if nplots%3==0 and ncol is None:
        ncol=3
    if ncol is None: ncol=4
    nrows = int(np.ceil(nplots/ncol))

    if width_ratios!=None or height_ratios!=None: use_subplots=True
    if width_ratios==None: width_ratios=np.repeat(1,ncol)
    if height_ratios==None: height_ratios=np.repeat(1,nrows)

    if figsize is None:
        figsize = (7*ncol*width_scale,height_scale*6*int(np.ceil(nplots/ncol)))
    regular_version = False
    if use_subplots:
        if 'figsize' in kwargs.keys():
            fig, ax = plt.subplots(nrows,ncol,**kwargs,
                            gridspec_kw={'width_ratios':width_ratios,'height_ratios':height_ratios})
        else:
            fig, ax = plt.subplots(nrows,ncol,figsize=figsize,
                            gridspec_kw={'width_ratios':width_ratios,'height_ratios':height_ratios},**kwargs)
        if 'dpi' in kwargs.keys(): fig.set_dpi(kwargs['dpi'])
    else:
        if 'figsize' in kwargs.keys():
            fig = plt.figure(**kwargs)
        else:
            fig = plt.figure(**kwargs, figsize = figsize)
        ax = []
        for i in np.arange(1,int(np.ceil(nrows*ncol))+1):
            ax += [fig.add_subplot(nrows, ncol, i)]
    return fig,np.array(ax).flatten()

def add_labels(x,y,ax):
    '''Add labels to scatter plot for each common index in series x and y.'''
    for i in np.intersect1d(x.index,y.index):
        ax.text(x.loc[i],y.loc[i],i,ha='center',va='top')

def twinx2(ax1,tw,n=2):
    '''Primary axis, secondary axis, number of digits to round to (default 2),
    does not return anything.'''
    l = ax1.get_ylim()
    l2 = tw.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax1.get_yticks())
    tw.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    tw.grid(None)
    tw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.'+str(n)+'f'))

def kstest100(x):
    '''Takes in series, creates 100 simulated normal
    distributions from the series mean, std, and length,
    and returns the mean coefficient and p-value of the
    Kolmogorov-Smirnov test of the series x and its
    simulated counterpart.'''
    coef, pval = [], []
    for n in np.arange(0,100):
        x_sim = stats.norm.rvs(loc=x.mean(),scale=x.std(),size=len(x),random_state=n)
        result = stats.kstest(x,x_sim)
        coef += [result[0]]
        pval += [result[1]]
    return np.mean(coef),np.mean(pval)

def add_regression_text(m,x,y,loc='bottom right',ax=0):
    '''
    adds the regression line equation to a plot of y vs x for model m.
    m = sm.GLS() model,
    x = pd.Series() or pd.DataFrame() of the independent variable(s)
    y = pd.Series() or pd.DataFrame() of the dependent variable
    loc = text location: bottom right, lower right, upper left, or upper right
    ax = axes on which to add the regression text
    '''
    n0 = 'e' if abs(m.params[0]) < 1e-2 else 'f'
    if len(m.params.index) > 1:
        n1 = 'e' if abs(m.params[1]) < 1e-2 else 'f'
        plus_or_minus = '+' if m.params['const'] > 0 else '-'
    else:
        n1 = 'e' if abs(m.params[0]) < 1e-2 else 'f'

    if type(ax)==int:
        if 'const' in m.params.index:
            if loc=='bottom right':
                plt.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                plt.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                plt.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                plt.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')
        else:
            if loc=='bottom right':
                plt.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                plt.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                plt.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                plt.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')
    else:
        if 'const' in m.params.index:
            if loc=='bottom right':
                ax.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                ax.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                ax.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                ax.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')
        else:
            if loc=='bottom right':
                ax.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[0],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                ax.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[0],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                ax.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[0],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                ax.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[0],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')

def find_best_dist(stacked_df, plot=True, print_chi_squared=False, bins=40, ax=0, density=False):
    '''takes a stacked dataframe and outputs a list
    of distributions that best fits that data (descending).
    stacked_df: pandas dataframe or series
    plot: bool, whether to plot the given data and
       the simulated data from best dist.
    print_chi_squared: bool, whether to print
       distribution results (minimize chi sq)
    bins: int, number of bins in histogram
    ax: matplotlib axis
    density: bool, whether to use length of data
       to form simulated distribution or to use
       1000 points and a normalized histogram
    '''
    y = stacked_df.copy()
    x = stacked_df.copy()
#     y = y[y>0]
    dist_names = ['weibull_min','norm','weibull_max','beta','invgauss','uniform','gamma','expon','lognorm','pearson3','triang','powerlognorm','logistic']
    chi_square_statistics = []
    # 11 equi-distant bins of observed Data
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y, percentile_bins)
    observed_frequency, hist_bins = (np.histogram(y, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(stats, distribution)
        try:
            param = dist.fit(y)
        except:
            param = [0,1]
#         print("{}\n{}\n".format(dist, param))


        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * y.size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square_statistics.append(ss)


    #Sort by minimum chi-square statistics
    results = pd.Series(chi_square_statistics,dist_names).sort_values()

    if print_chi_squared:
        print ('\nDistributions listed by goodness of fit:')
        print ('.'*40)
        print (results)

    if results.notna().any():
        best_dist = getattr(stats,results.idxmin())
    else:
        best_dist = getattr(stats,'anglit')
#     print(results.idxmin())
    best_params = best_dist.fit(y)
#     print(best_params)
    best_sim_size = 1000 if density else len(y)
    if len(best_params)==2:
        best_sim = best_dist.rvs(best_params[0],best_params[1],size=len(y),random_state=0)
    elif len(best_params)==3:
        best_sim = best_dist.rvs(best_params[0],best_params[1],best_params[2],size=len(y),random_state=0)
    elif len(best_params)==4:
        best_sim = best_dist.rvs(best_params[0],best_params[1],best_params[2],best_params[3],size=len(y),random_state=0)

    if plot:
        if type(ax)==int:
            fig,ax = easy_subplots(1,1)
            ax = ax[0]
        ax.hist(x.values.flatten(),bins=np.linspace(y.min(),y.max(),bins),color='tab:blue',alpha=0.5,density=density)
        ax.hist(best_sim,bins=np.linspace(y.min(),y.max(),bins),color='tab:orange',alpha=0.5,density=density)
        ax.set(title=results.index[0])
#         ax[1].plot(y,best_dist.pdf(y,))
    return results.index

def year_decimal_to_datetime(year_decimal):
    '''
    Takes in any year-as-decimal value and returns
    the equivalent (to microsecond) datetime form.
    Can be used to convert an entire index, i.e.
    df.index = [year_decimal_to_datetime(j) for j in df.index]
    '''
    n_years = np.floor(year_decimal)
    months = (year_decimal-n_years)*12+1
    n_months = np.floor(months)
    days_in_month = 31 if (n_months%2!=0 and n_months<8) or (n_months%2==0 and n_months>=8) else 30 if n_months!=2 else 28 if n_years%4!=0 else 29
    days = (months-n_months)*days_in_month+1
    n_days = np.floor(days)
    hours = (days-n_days)*24
    n_hours = np.floor(hours)
    minutes = (hours-n_hours)*60
    n_minutes = np.floor(minutes)
    seconds = (minutes-n_minutes)*60
    n_seconds = np.floor(seconds)
    microseconds = (seconds-n_seconds)*1000
    n_microseconds = np.floor(microseconds)
    return datetime(int(n_years), int(n_months), int(n_days), int(n_hours), int(n_minutes), int(n_seconds), int(n_microseconds))

def pval_to_star(pval, no_star_cut=0.1, period_cut=0.05, one_star_cut=0.01, two_star_cut=0.001):
    """
    Converts a value from its numerical value to a string where:
    *** < 0.001 < ** < 0.01 < * < 0.05 < . < 0.1
    """
    pval_str = '***' if pval < two_star_cut else '**' if pval < one_star_cut else '*' if \
        pval < no_star_cut else '(.)' if pval < period_cut else ''
    return pval_str

def profile_with_cpython(evaluation_string, filename='compile_time'):
    if True:
        pr = cProfile.Profile()
        pr.enable()
        exec(evaluation_string)
        pr.disable()
        s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        with open(filename, 'w+') as f:
            f.write(s.getvalue())

def add_axis_labels(fig, option='ylabel_width', xloc=-0.1, yloc=1.03, zloc=None, skip=None, alt_xlocs=False, start_letter=0):
    """
    Adds lettering to each figure for scientific publication figures.
    Option input can be any in [`price_middle_column`,`subset1`,`all`,`subset2`],
    otherwise uses the xloc and yloc values given. Option inputs allow for
    changes based on xtick characteristics

    price_middle_column: largest left offset from y axis
    subset1: smallest offset
    all: equivalent to subset1
    subset2: moderate offset
    """
    skip = [skip] if type(skip)==int else skip if skip is not None else []
    chars = character_list[10:36][start_letter:]
    ax = fig.axes
    ax = [a for e,a in enumerate(ax) if e not in skip]
    if alt_xlocs:
        xlocs     = {1:-0.09, 2:-0.12, 3:-0.15, 4:-0.18, 5:-0.21, 6:-0.24, 7:-0.27, 8:-0.30}
        xlocs_dec = {1:-0.11, 2:-0.14, 3:-0.17, 4:-0.20, 5:-0.23, 6:-0.26, 7:-0.29, 8:-0.32}
    else:
        xlocs     = {1:-0.06, 2:-0.12, 3:-0.15, 4:-0.18, 5:-0.21, 6:-0.24, 7:-0.27, 8:-0.30}
        xlocs_dec = {1:-0.11, 2:-0.14, 3:-0.17, 4:-0.20, 5:-0.23, 6:-0.26, 7:-0.29, 8:-0.32}
    for label,a in zip(chars, ax):
        xticks = len(a.get_xticks())
        ylabel_width = max([len(str(i).split("'")[1].split("'")[0]) for i in a.get_yticklabels()])
        ylabel_width_no_dec = max([len(str(i).split("'")[1].split("'")[0].replace('.','')) for i in a.get_yticklabels()])
        has_dec = ylabel_width!=ylabel_width_no_dec
        if option=='ylabel_width':
            xloc = xlocs_dec[ylabel_width_no_dec] if has_dec else xlocs[ylabel_width]
        elif option=='price_middle_column':
            xloc = -0.21 if a in ax[1::3] else -0.17
            yloc = 1.02
        elif option in ['subset1','all']:
            xloc = -0.035 if xticks>10 else -0.07 if xticks>7 else -0.08
            yloc = 1.03
        elif option=='subset2':
            xloc = -0.05 if xticks>6 else -0.08 if xticks>4 else -0.13 if xticks>3 else -0.18
            yloc = 1.03
        
        if zloc is None:
            a.text(xloc,yloc,label+')', transform=a.transAxes)
        else:
            try:
                a.text2D(xloc,yloc,label+')', transform=a.transAxes)
            except:
                a.text(xloc,yloc,label+')', transform=a.transAxes)

def get_line_intersection(line1_point1, line1_point2, line2_point1, line2_point2):
    """
    Finds the intersection point between two lines defined by two points each.

    Args:
        point1 (tuple): First point (x1, y1) on the first line.
        point2 (tuple): Second point (x2, y2) on the first line.

    Returns:
        tuple: Intersection point (x, y) of the two lines.
    """
    x1, y1 = line1_point1
    x2, y2 = line1_point2

    # Calculate the slope of the first line
    slope1 = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept of the first line
    intercept1 = y1 - slope1 * x1

    x1, y1 = line2_point1
    x2, y2 = line2_point2

    # Calculate the slope of the second line
    slope2 = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept of the second line
    intercept2 = y1 - slope2 * x1

    # Calculate the x-coordinate of the intersection point
    x_intersection = (intercept1 - intercept2) / (slope2 - slope1)
    
    # Calculate the y-coordinate of the intersection point
    y_intersection = slope1 * x_intersection + intercept1

    return x_intersection, y_intersection


def get_surface_intersections(surface1, surface2):
    """
    Surface 1 and Surface 2 should each be in the form (x,y,z) that is capable of being plotted using
    ax.plot_trisurf, and should share the same x and y values. Returns array of intersection values
    in the same (x,y,z) coordinate system, where ax.plot(x,y,z) should give the intersection line. 
    """
    inter_prices = np.unique(surface1[1])
    intersections = np.zeros([inter_prices.shape[0],3])
    for i,inter_price in enumerate(inter_prices):
        inter = surface1[1]==inter_price
        price_inter = surface1[0][inter]
        prod_inter  = surface1[2][inter]
        prod_inter_d  = surface2[2][inter]
        rmse_inter = prod_inter-prod_inter_d
        rmse_inter1 = rmse_inter[rmse_inter>0]
        rmse_inter2 = rmse_inter[rmse_inter<0]    
        if len(rmse_inter1)==0 or len(rmse_inter2)==0:
            rmse_inter = ((prod_inter-prod_inter_d)**2)
            min_rmse_idx_inter1 = np.nanargmin(rmse_inter)
            rmse_inter[rmse_inter==rmse_inter[min_rmse_idx_inter1]] = np.inf
            min_rmse_idx_inter2 = np.nanargmin(rmse_inter)
        else:
            min_rmse_idx_inter1 = np.nanargmax(np.min(rmse_inter1)==rmse_inter)
            min_rmse_idx_inter2 = np.nanargmax(np.max(rmse_inter2)==rmse_inter)
            # min_rmse_idx_inter1 = np.nanargmin(rmse_inter1)
            # min_rmse_idx_inter2 = np.nanargmax(rmse_inter2)

        x = price_inter[min_rmse_idx_inter1], price_inter[min_rmse_idx_inter2]
        if x[0]==x[1]:
            x = price_inter[min_rmse_idx_inter1], price_inter[min_rmse_idx_inter2]+1e-9
        y1 = prod_inter[min_rmse_idx_inter1], prod_inter[min_rmse_idx_inter2]
        y2 = prod_inter_d[min_rmse_idx_inter1], prod_inter_d[min_rmse_idx_inter2]
        intersect = get_line_intersection([x[0],y1[0]], [x[1], y1[1]], [x[0],y2[0]], [x[1], y2[1]])
        intersections[i] = [intersect[0], inter_price, intersect[1]] # (price_0, pirce_1, production)
    return intersections

def try_float_conversion(string):
    try:
        return float(string)
    except:
        return string

def str_to_dict(string):
    new_string = re.sub(r"\{*\}*",'',string)
    str_list = re.findall("'\w+':\s*[0-9a-zA-z-.]*",new_string)
    new_dict = {i.split(':')[0].replace('"','').replace("'",''): try_float_conversion(''.join(i.split(':')[1:]).replace('"','').replace("'",'')) for i in str_list}
    return new_dict

def AIC_linearmodels(panel_model):
    """ 
    For use with linearmodels panel regression models.
    
    Eqn from https://www.statology.org/aic-in-python/
    """
    L=panel_model.loglik
    K=panel_model.df_model+2
    return 2*K - 2*L

def hausman_linearmodels(fe, re):
    """
    Input fe, re
    
    If p<0.05, should use fixed effects

    Compute hausman test for fixed effects/random effects models
    b = beta_fe
    B = beta_re
    From theory we have that b is always consistent, but B is consistent
    under the alternative hypothesis and efficient under the null.
    The test statistic is computed as
    z = (b - B)' [V_b - v_B^{-1}](b - B)
    The statistic is distributed z \sim \chi^2(k), where k is the number
    of regressors in the model.
    Parameters
    ==========
    fe : statsmodels.regression.linear_panel.PanelLMWithinResults
        The results obtained by using sm.PanelLM with the
        method='within' option.
    re : statsmodels.regression.linear_panel.PanelLMRandomResults
        The results obtained by using sm.PanelLM with the
        method='swar' option.
    Returns
    =======
    chi2 : float
        The test statistic
    df : int
        The number of degrees of freedom for the distribution of the
        test statistic
    pval : float
        The p-value associated with the null hypothesis
    
    Notes
    =====
    The null hypothesis supports the claim that the random effects
    estimator is "better". If we reject this hypothesis it is the same
    as saying we should be using fixed effects because there are
    systematic differences in the coefficients.
    
    Tests whether random effects estimator can be used, since it is 
    more efficient but can be biased, so if the fixed and random 
    effects estimators are not equal, the fixed effects estimator 
    is the correct/consistent one.
    
    If p<0.05, should use fixed effects
    """
    
    import numpy.linalg as la
    
    # Pull data out
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov

    # NOTE: find df. fe should toss time-invariant variables, but it
    #       doesn't. It does return garbage so we use that to filter
    df = b[np.abs(b) < 1e8].size

    # compute test statistic and associated p-value
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
    pval = stats.chi2.sf(chi2, df)

    return chi2, df, pval

def pesaran_cd_test(model):
    """
    Implemented as described in section 2.1 of Hoyos and Sarafidis, "Testing for cross-sectional dependence in panel-data models", The Stata Journal (2006).
    https://journals.sagepub.com/doi/pdf/10.1177/1536867X0600600403

    Requires that the model have the attribute `resids`, and that it is a pandas Series with 2-level multi-index where the second level is the time index.

    Model must be balanced (can add unbalanced functionality using the paper cited above, there is a method)

    Returns (coefficient, pvalue)
    """
    u_hat = model.resids.unstack().values
    N,T = u_hat.shape
    i_size = N
    j_size = i_size-1

    rho_hat = np.zeros([i_size,i_size])
    for i in range(i_size):
        j_inds = [k for k in range(i_size) if k!=i]
        u_i = u_hat[i,:]
        for j in j_inds:
            u_j = u_hat[j,:]
            rho_hat[i,j] = np.sum(u_i*u_j) / \
                ((np.sum(u_i**2)**0.5) * (np.sum(u_j**2)**0.5))
    CD = ((2*T)/(N*(N-1)))**0.5 * np.sum([np.sum(rho_hat[i,i+1:]) for i in range(i_size-1)])
    pvalue = stats.norm.sf(abs(CD))*2
    return CD, pvalue
        
def convert_compare_to_df(result_dict, parens='std_errors'):
    '''
    Converts the outputs of the linearmodels.panel.compare method into a pandas dataframe so it can be more easily copied and/or saved

    result_dict: dict, with keys the names of the models and values the models being compared
    parens: str, options are std_errors or t_stats, changes the values displayed in the parentheses; default is standard errors
    '''
    comp = compare(result_dict)
    properties = {
        'estimator_method': 'Estimator',
        'nobs': 'No. Observations',
        'cov_estimator': 'Cov. Est.',
        'rsquared': 'R-Squared',
        'rsquared_within': 'R-Squared (Within)',
        'rsquared_between': 'R-Squared (Between)',
        'rsquared_overall': 'R-Squared (Overall)',
        'f_statistic': 'F-statistic',
        'params': 'Parameters',
        'pvalues': 'P-values',
        'std_errors': 'Standard Errors',
    }
    prop_keys = [s for s in properties.keys() if len(getattr(comp,s).shape)<=1 or getattr(comp,s).shape[1]<2]
    outputs = pd.concat([getattr(comp,s) for s in prop_keys], keys=prop_keys, axis=1).T.rename(properties)
    for r in [i for i in outputs.index if 'R-Sq' in i]:
        outputs.loc[r,:] = ['{:.3f}'.format(k) for k in outputs.loc[r].values]
    params = comp.params.copy()
    std_errors = comp.std_errors if parens=='std_errors' else comp.tstats
    pvals = comp.pvalues
    for i in params.index:
        for c in params.columns:
            params.loc[i,c] = '' if params.isna()[c][i] else '{:.3f}{:s}\n({:.3f})'.format(params[c][i], pval_to_star(pvals[c][i]), std_errors[c][i])
    if hasattr(params.index,'get_level_values'):
        params.index = [f'{i[1]} {i[0]}' for i in params.index]
    fstat = comp.f_statistic.T.rename({'F stat':'F-statistic','P-value':'P-value (F-stat)'})
    fstat.loc['F-statistic', :] = ['{:.2f}'.format(i) for i in fstat.loc['F-statistic', :].values]
    fstat.loc['P-value (F-stat)', fstat.loc['P-value (F-stat)']<1e-6] = ['{:.3e}'.format(i) for i in fstat.loc['P-value (F-stat)', fstat.loc['P-value (F-stat)']<1e-6].values]
    effects = pd.DataFrame([','.join(k.included_effects) for k in result_dict.values()], index=result_dict.keys(), columns=['Effects']).T
    outputs = pd.concat([
        pd.DataFrame(outputs.columns, columns=['Dep. Variable'], index=outputs.columns).T,
        outputs,
        fstat,
        params,
        effects
    ])
    return outputs

init_plot2()
