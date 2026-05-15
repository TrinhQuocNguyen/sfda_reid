import numpy as np


def _get_plt():
    import matplotlib.pyplot as plt

    return plt


def _project_2d(features):
    try:
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(features)
    except ImportError:
        centered = features - np.mean(features, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T

def plot_tsne(features, labels, cam_ids, save_path, title):
    plt = _get_plt()
    emb = _project_2d(features)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab20', marker='o', s=10, alpha=0.7)
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_retrieval_examples(query_img, gallery_imgs, gallery_labels, query_pid, save_path):
    plt = _get_plt()
    import matplotlib.patches as patches
    fig, axes = plt.subplots(1, len(gallery_imgs) + 1, figsize=(18, 3), dpi=300)
    axes[0].imshow(query_img)
    axes[0].set_title('Query')
    axes[0].axis('off')
    for i, (img, label) in enumerate(zip(gallery_imgs, gallery_labels)):
        axes[i+1].imshow(img)
        color = 'green' if label == query_pid else 'red'
        rect = patches.Rectangle((0,0),img.shape[1],img.shape[0],linewidth=5,edgecolor=color,facecolor='none')
        axes[i+1].add_patch(rect)
        axes[i+1].axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_bound_vs_empirical(epochs, bounds, empirical_errors, save_path):
    plt = _get_plt()
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(epochs, bounds, label='Theoretical Bound')
    plt.plot(epochs, empirical_errors, label='Rank-1 Error')
    plt.fill_between(epochs, bounds, empirical_errors, color='gray', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_ablation_bar(results_dict, save_path):
    plt = _get_plt()
    labels = list(results_dict.keys())
    mAPs = [v['mAP'] for v in results_dict.values()]
    rank1s = [v['rank1'] for v in results_dict.values()]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6), dpi=300)
    plt.bar(x - width/2, mAPs, width, label='mAP')
    plt.bar(x + width/2, rank1s, width, label='Rank-1')
    plt.xticks(x, labels, rotation=15)
    plt.ylabel('Score (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
