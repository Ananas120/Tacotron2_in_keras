import numpy as np
import matplotlib.pyplot as plt

def plot(y, figsize=(7,4), axe_x=None, 
         titre="Titre", xlabel="", ylabel="", 
         fontsize=22., labelsize=20., 
         xmin=0, xmax=None,
         linewidth=2., tick_label=None,
         type_graph='plot', filename=None, show=False):
    if xmax is None: xmax = len(y)
    if axe_x is None: axe_x = np.linspace(xmin, xmax, len(y))
    fig = plt.figure(figsize=figsize)
    plt.xlim(xmin, xmax)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.title(titre, fontsize=fontsize)
    plt.tick_params(axis='both',labelsize=labelsize)
    if type_graph == 'plot':
        plt.plot(axe_x, y, linewidth=linewidth)
    elif type_graph == 'bar':
        plt.bar(axe_x, y, tick_label=tick_label)
    elif type_graph == 'img':
        plt.imshow(y)
    else:
        print("Type de graphique :",type_graph,"non géré !")
    if filename is not None:
        plt.savefig(filename)
        if show: plt.show()
    else:
        plt.show()
    plt.close()
    
def plot_multiple(liste_y, liste_labels=None, figsize=(7,4),
                  titre="Titre", xlabel="", ylabel="", 
                  fontsize=22., labelsize=20., 
                  xmin=0, xmax=None,
                  linewidth=2., tick_label=None,
                  type_graph='plot', loc="upper left", legend_fontsize=16, 
                  filename=None, show=False):
    liste_label_y = None
    if type(liste_y) == dict:
        liste_label_y = [(label, y) for label, y in liste_y.items()]
    elif liste_labels is not None and len(liste_y) == len(liste_labels):
        liste_label_y = [(label, y) for label, y in zip(liste_labels, liste_y)]
    else:
        print("Erreur lors de l'affichage !")
        return
    fig = plt.figure(figsize=figsize)
    plt.title(titre, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(axis='both',labelsize=labelsize)
    if type_graph == 'plot':
        for label, y in liste_label_y:
            plt.plot(y, linewidth=linewidth, label=label)
    else:
        print("Type de graphique :",type_graph,"non géré !")
    plt.legend(loc=loc, fontsize=legend_fontsize)
    if filename is not None:
        plt.savefig(filename)
        if show: plt.show()
    else:
        plt.show()
    plt.close()

def plot_spectrogram(spectrogram, auto_aspect=True, figsize=(10, 8), fontsize=30, filename=None, show=False):
    nrows = 2 if target_spectrogram is not None else 1
    fig, ax = plt.subplots(nrows=nrows, figsize=figsize)
    
    if nrows == 1: ax = [ax]
    
    im = ax[0].imshow(np.rot90(spectrogram), aspect='auto', interpolation='none')
    ax[0].set_title(title, fontsize=fontsize)
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax[0])
    
    if target_spectrogram is not None:
        im2 = ax[1].imshow(np.rot90(target_spectrogram), aspect='auto', interpolation='none')
        ax[1].set_title("Target spectrogram", fontsize=fontsize)
        fig.colorbar(mappable=im2, shrink=0.65, orientation='horizontal', ax=ax[1])

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        if show: plt.show()
    else:
        plt.show()
    plt.close()

def plot_alignment(alignment, filename=None, title=None, fontsize=30, max_len=None, show=False, figsize=(8,8)):
    if max_len is not None:
        alignment = alignment[:, :max_len]

    fig = plt.figure(figsize=figsize)
    
    im = plt.imshow(alignment,
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    fig.colorbar(im)

    plt.xlabel('Decoder timestep', fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.ylabel('Encoder timestep', fontsize=fontsize)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(path, format='png')
        if show: plt.show()
    else:
        plt.show()
    plt.close()
