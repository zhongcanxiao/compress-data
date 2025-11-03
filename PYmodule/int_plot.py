import numpy as np

import matplotlib.pyplot as plt

from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


def int_plot_error(slider_vars,lossdata5d):
    [mask_min,mask_max,mask_step]=slider_vars['mask']
    [count_min,count_max,count_step]=slider_vars['count']
    [variance_min,variance_max,variance_step]=slider_vars['variance']
    
    pltymax=np.max(lossdata5d[:,:,:,:,4])
    pltymin=np.min(lossdata5d[:,:,:,:,4])
    if(int((mask_max-mask_min)/mask_step)!=lossdata5d.shape[0]) or\
      (int((count_max-count_min)/count_step)!=lossdata5d.shape[0]) or\
      (int((variance_max-variance_min)/variance_step)!=lossdata5d.shape[0]) :
        exit("slider range not compatible with plot data dimension")
    def plot_error(mask,count,variance):
        #[idx_mask,idx_count,idx_variance,idx_scalar]=numlist
        
        idx_mask=int((mask-mask_min)/mask_step)
        idx_count=int((count-count_min)/count_step)
        idx_variance=int((variance-variance_min)/variance_step)
        #idx_scalar=int(scalar/0.01-220)
        #print(idx_mask,idx_count,idx_variance,idx_scalar)
        pltdata = lossdata5d[idx_mask,idx_count,idx_variance,:,3:5]
        plt.figure(figsize=(10, 6))
        plt.plot(pltdata[:,0],pltdata[:,1])

        plt.title('Performance with Hyper parameter tuning tuning stopping criteria')
        plt.xlabel('scaling factor of sample measurement data')
        plt.ylabel('MAE between depeaked scaled sample measurement and KD tree predicted background')
        plt.ylim(pltymin,pltymax)
        plt.grid(True)
        plt.show()
    # Create an interactive slider
    mask_slider = widgets.FloatSlider(value=0, min=mask_min, max=mask_max, step=mask_step, description='Mask number ')
    count_slider = widgets.FloatSlider(value=0, min=count_min, max=count_max, step=count_step, description='Count')
    variance_slider = widgets.FloatSlider(value=0, min=variance_min, max=variance_max, step=variance_step, description='Variance')
    #scalar_slider = widgets.FloatSlider(value=2, min=2.2, max=2.79, step=0.01, description='Scalar')


    # Use the interact function to update the plot
    interact(plot_error, mask=mask_slider, count=count_slider, variance=variance_slider)#, scalar=scalar_slider)

def int_sliceviewer(data3d,**kwargs):
    xmin=kwargs.get('xmin',None)
    ymin=kwargs.get('ymin',None)
    zmin=kwargs.get('zmin',None)
    xmax=kwargs.get('xmax',None)
    ymax=kwargs.get('ymax',None)
    zmax=kwargs.get('zmax',None)
    if xmin == None: xmin=0
    if xmax == None: xmax=data3d.shape[0]
    if ymin == None: ymin=0
    if ymax == None: ymax=data3d.shape[1]
    if zmin == None: zmin=0
    if zmax == None: zmax=data3d.shape[2]

    pos =kwargs.get('pos', None)
    cbmax_init =kwargs.get('cbmax', None)
    axis_init =kwargs.get('axis', None)
    

    xpos=kwargs.get('xpos',None)
    ypos=kwargs.get('ypos',None)
    zpos=kwargs.get('zpos',None)
    def plot_slice(axis, index,cbmax,scale_mode):
        plt.figure(figsize=(6, 6))
        if(scale_mode=='log' or scale_mode=='logit'):
            cbmin=0.1
        else:
            cbmin=np.min(data3d)
        if axis == 'X':
            im=plt.imshow(data3d[index, :,:], cmap='viridis',vmin=cbmin,vmax=cbmax,norm=scale_mode)
            plt.title(f'X-axis slice at index {index}')
            plt.ylim(ymin,ymax)
            plt.xlim(zmin,zmax)
            plt.xlabel('Z')
            plt.ylabel('Y')        
        elif axis == 'Y':
            im=plt.imshow(data3d[:, index,:], cmap='viridis',vmin=cbmin,vmax=cbmax,norm=scale_mode)
            plt.title(f'Y-axis slice at index {index}')
            plt.ylim(xmin,xmax)
            plt.xlim(zmin,zmax)
            plt.xlabel('Z')
            plt.ylabel('X')
        elif axis == 'Z':
            im=plt.imshow(data3d[:, :, index], cmap='viridis',vmin=cbmin,vmax=cbmax,norm=scale_mode)
            plt.title(f'Z-axis slice at index {index}')
            plt.ylim(xmin,xmax)
            plt.xlim(ymin,ymax)
            plt.xlabel('Y')
            plt.ylabel('X')
        plt.colorbar(im,fraction=0.0457,pad=0.04)
        

    
        
        plt.show()

    # Create widgets
    [xsize,ysize,zsize]=data3d.shape
    if axis_init is None: axis_init='X'
    axis_widget = widgets.Dropdown(options=['X', 'Y', 'Z'], value=axis_init,description='Axis:')
    scale_mode_widget = widgets.Dropdown(options=["linear", "log"], description='Scale Mode:')
    #scale_mode_widget = widgets.Dropdown(options=["linear", "log", "symlog", "logit"], description='Scale Mode:')

    if pos is not None: 
        index_pos=pos
    else:
        index_pos=0
    if cbmax_init is None: cbmax_init=np.max(data3d)
    cbmax_widget = widgets.IntSlider(min=0,max=np.max(data3d), value=cbmax_init,description='Color Bar Range Max:')
    index_widget = widgets.IntSlider(min=0, max=data3d.shape[0] - 1, value=index_pos,step=1, description='Index:')
    
    # Update function for the slider range
    def update_slider_range(*args):
        if axis_widget.value == 'X':
            index_widget.max = data3d.shape[0] - 1
            if xpos is not None: index_widget.value = xpos
        elif axis_widget.value == 'Y':
            index_widget.max = data3d.shape[1] - 1
            if ypos is not None: index_widget.value = ypos
        elif axis_widget.value == 'Z':
            index_widget.max = data3d.shape[2] - 1
            if zpos is not None: index_widget.value = zpos

    # Attach the update function to the dropdown change event
    axis_widget.observe(update_slider_range, 'value')


    widgets.interact(plot_slice, axis=axis_widget, index=index_widget,cbmax=cbmax_widget,scale_mode=scale_mode_widget)

##########################################################################################################################################
# 3D
##########################################################################################################################################
import plotly.graph_objects as go

def int_plot3d(plot_data,plot_mask):
    def plot_3d(xmin,xmax,ymin,ymax,zmin,zmax):
        data_plt=[]
        data_trim=plot_data[xmin:xmax,ymin:ymax,zmin:zmax]
        plot_mask_trim=plot_mask[xmin:xmax,ymin:ymax,zmin:zmax]
        
        for index, x in np.ndenumerate(data_trim):
            
            if(plot_mask_trim[index]):
                data_plt.append([list(index)+[np.log10(x) ]])
        sp=np.array(data_plt)
        print(sp.shape)


        fig = go.Figure(data=[go.Scatter3d(
            x=sp[:,0,0],
            y=sp[:,0,1],
            z=sp[:,0,2],
            mode='markers',
            marker=dict(
                size=2,
                color=sp[:,0,3],                # set color to an array/list of desired values
                colorscale='rainbow',   # choose a colorscale
                #colorscale='viridis',   # choose a colorscale
                opacity=0.3
            )
        )])

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=1))
        fig.show()



    xmin_widget = widgets.IntSlider(min=0, max=500, step=1, value=200, description='xmin:')
    ymin_widget = widgets.IntSlider(min=0, max=500, step=1, value=210, description='ymin:')
    zmin_widget = widgets.IntSlider(min=0, max=500, step=1, value=200, description='zmin:')

    xmax_widget = widgets.IntSlider(min=20, max=500, step=1, value=230, description='xmax:')
    ymax_widget = widgets.IntSlider(min=20, max=500, step=1, value=240, description='ymax:')
    zmax_widget = widgets.IntSlider(min=20, max=500, step=1, value=230, description='zmax:')

    widgets.interact(plot_3d, xmin=xmin_widget,xmax=xmax_widget,
                    ymin=ymin_widget,ymax=ymax_widget,
                    zmin=zmin_widget,zmax=zmax_widget                 )
    

import plotly.graph_objects as go
import numpy as np


from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display

from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.cm as cm

InteractiveShell.ast_node_interactivity = "all"

def int_plot3d_alpha(plot_data,plot_mask):
    def plot_3d(xmin,xmax,ymin,ymax,zmin,zmax):
        data_plt=[]
        data_trim=plot_data[xmin:xmax,ymin:ymax,zmin:zmax]
        plot_mask_trim=plot_mask[xmin:xmax,ymin:ymax,zmin:zmax]
        
        for index, x in np.ndenumerate(data_trim):
            
            if(plot_mask_trim[index]):
                data_plt.append([list(index)+[x ]])
        sp=np.array(data_plt)
        print(sp.shape)
        pltdatamax=np.max(sp[:,0,3])
        print(pltdatamax)

        num_opacity=17
        viridis = plt.cm.get_cmap('viridis', 256)
        norm = plt.Normalize(vmin=np.min(sp[:,0,3]), vmax=np.max(sp[:,0,3]))
        viridis = cm.get_cmap('viridis')
        rgba_values = viridis(norm(sp[:,0,3]))
        #plt_data_percentiles = np.percentile(sp[:,0,3], range(0,100,100/num_opacity))
        plt_data_percentiles = np.percentile(sp[:,0,3], [10,40,60,70,90,95,98])
        print(plt_data_percentiles)

        for i in range(len(sp[:,0,3])):
          rgba_values[i][3]=0.
          for critical_value in plt_data_percentiles:
            if (sp[:,0,3][i] > critical_value): rgba_values[i][3]+=1.0/num_opacity
        rgba_values[:][3]=.00
                    
        fig = go.Figure(data=[go.Scatter3d(
            x=sp[:,0,0],
            y=sp[:,0,1],
            z=sp[:,0,2],
            mode='markers',
            marker=dict(
                size=3,
                color=rgba_values
            )
        )])

        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=1))
        fig.show()



    # peak2
    xmin_widget = widgets.IntSlider(min=0, max=500, step=1, value=165, description='xmin:')
    ymin_widget = widgets.IntSlider(min=0, max=500, step=1, value=210, description='ymin:')
    zmin_widget = widgets.IntSlider(min=0, max=500, step=1, value=95, description='zmin:')

    xmax_widget = widgets.IntSlider(min=20, max=500, step=1, value=180, description='xmax:')
    ymax_widget = widgets.IntSlider(min=20, max=500, step=1, value=230, description='ymax:')
    zmax_widget = widgets.IntSlider(min=20, max=500, step=1, value=115, description='zmax:')
    
    # peak1
    ymin_widget = widgets.IntSlider(min=0, max=500, step=1, value=215, description='ymin:')
    zmin_widget = widgets.IntSlider(min=0, max=500, step=1, value=200, description='zmin:')
    xmin_widget = widgets.IntSlider(min=0, max=500, step=1, value=200, description='xmin:')

    xmax_widget = widgets.IntSlider(min=20, max=500, step=1, value=230, description='xmax:')
    ymax_widget = widgets.IntSlider(min=20, max=500, step=1, value=240, description='ymax:')
    zmax_widget = widgets.IntSlider(min=20, max=500, step=1, value=230, description='zmax:')



    widgets.interact(plot_3d, xmin=xmin_widget,xmax=xmax_widget,
                    ymin=ymin_widget,ymax=ymax_widget,
                    zmin=zmin_widget,zmax=zmax_widget                 )
