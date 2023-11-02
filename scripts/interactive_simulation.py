import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from random import uniform
import scipy.ndimage as ndimage
from tqdm import tqdm
import scipy.optimize as opt

from main import Percolation2D

def sliders_on_changed(val):
    lattice, labeled_lattice, max_cluster = percolation2d.generate(size, prob_slider.val)
    ax1_obj.set_data(lattice)
    ax2_obj.set_data(max_cluster)

    fig.canvas.flush_events()
    
def reset_button_on_clicked(mouse_event):
    prob_slider.reset()
    reset_button.on_clicked(reset_button_on_clicked)

if __name__ == '__main__':


    percolation2d = Percolation2D() #create class object

    p_c = 0.59274605079210 #transition prob
    size = 250

    fig,(ax1,ax2) = plt.subplots(1,2)    #subplot
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    
    fig.subplots_adjust(left=0.25, bottom=0.25) # Adjust the subplots region to leave some space for the sliders and buttons

    
    prob_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])    #Define an axes area and draw a slider in it
    prob_slider = Slider(prob_slider_ax, 'Amp', 0.01, 1.0, valinit=p_c)
    reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])  # Add a button for resetting the parameters
    reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')

    p = prob_slider.val
    lattice, labeled_lattice, max_cluster = percolation2d.generate(size, p)   #generate lattice

    ax1_obj = ax1.imshow(lattice, cmap = "binary")
    ax2_obj = ax2.imshow(max_cluster, cmap= "binary")

    # Define an action for modifying the line when any slider's value changes
    prob_slider.on_changed(sliders_on_changed)
    reset_button.on_clicked(reset_button_on_clicked)
    
    plt.show()
