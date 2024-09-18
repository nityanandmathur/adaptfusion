import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
from tqdm import tqdm

def main() -> None:
    parser = argparse.ArgumentParser(description='Combine images')
    parser.add_argument('--source', type=str, required=True, help='Path to the source images')
    parser.add_argument('--none', type=str, required=True, help='Path to the No LoRA images')
    parser.add_argument('--full', type=str, required=True, help='Path to the Full LoRA images')
    parser.add_argument('--selective', type=str, required=True, help='Path to the Selective LoRA images')
    parser.add_argument('--output', type=str, required=True, help='Path to the output images')

    args = parser.parse_args()
    images_name = os.listdir(args.source)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    for name in tqdm(images_name):
        source_image = Image.open(os.path.join(args.source, name))
        # source_image = source_image.crop((224,62,1824,962))
        none_image = Image.open(os.path.join(args.none, name))
        full_image = Image.open(os.path.join(args.full, name))
        selective_image = Image.open(os.path.join(args.selective, name))

        fig, ax = plt.subplots(1, 4, figsize=(14, 2.7))  # Increase the figure size
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.1)  # Adjust margins and spacing

        ax[0].imshow(source_image)
        ax[0].set_title('Source')
        ax[0].axis('off')

        ax[1].imshow(none_image)
        ax[1].set_title('No LoRA')
        ax[1].axis('off')

        ax[2].imshow(full_image)
        ax[2].set_title('Full LoRA')
        ax[2].axis('off')

        ax[3].imshow(selective_image)
        ax[3].set_title('Selective LoRA')
        ax[3].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output, name))
        plt.close()

if __name__ == '__main__':
    main()
