import numpy as np
import matplotlib.pyplot as plt


def drawPillar():     
    n_groups = 1;       
    perdic42 = (0.45)
    perdic44 = (0.67)
    perdic82 = (0.33)    
         
    fig, ax = plt.subplots()    
    index = np.arange(n_groups)    
    bar_width = 0.35    
         
    opacity = 0.4    
    rects1 = plt.bar(index, perdic42, bar_width,alpha=opacity, color='b',label=    'cell 4*4 block 2*2')    
    rects2 = plt.bar(index + bar_width, perdic44, bar_width,alpha=opacity,color='r',label='cell 4*4 block 4*4')
    rects3 = plt.bar(index + bar_width*2, perdic82, bar_width, alpha=opacity, color='g', label="cell 8*8 block 2*2")    
         
    plt.xlabel('Cell and block size of HOG descriptor')    
    plt.ylabel('Prediction error')     
    plt.xticks(index + bar_width, ('parameter'))
    plt.legend()
    plt.tight_layout()
    plt.show()

drawPillar()