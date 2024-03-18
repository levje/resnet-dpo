import json
import matplotlib.pyplot as plt

def save_learn_hists(learn_hists, filename, title) -> None:
    learn_hists['title'] = title
    with open(filename, 'w') as f:
        json.dump(learn_hists, f)

def load_learn_hists(filename) -> dict:
    with open(filename, 'r') as f:
        return json.load(f)

def visualize_learn_hists(learn_hists: list[dict], save_path: str = None) -> None:
    n_hists = len(learn_hists)
    # For each learn_hist, plot the train and validation accuracy
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for lh in learn_hists:
        print(lh['title'])
        ax[0].plot(lh['train_acc'], label=lh['title'])
        ax[1].plot(lh['val_acc'], label=lh['title'])
        ax[2].plot(lh['train_loss'], label=lh['title'])
        ax[0].set_title('Train accuracy')
        ax[1].set_title('Validation accuracy')
        ax[2].set_title('Loss')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

    if save_path is not None:
        fig.savefig(save_path)
        print(f'Plot saved to {save_path}')
    else:
        print("Showing the plot...")
        plt.show()
