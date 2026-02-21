import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Match your STM32 defines
NUM_MELS = 80
NUM_FRAMES = 241
SPEC_SIZE = NUM_MELS * NUM_FRAMES

def load_spectrograms(path):
    """Load one or more spectrograms from a binary float32 file."""
    data = np.fromfile(path, dtype=np.float32)
    if data.size % SPEC_SIZE != 0:
        raise ValueError(
            f"File {path} has {data.size} floats, not divisible by {SPEC_SIZE}"
        )
    num_specs = data.size // SPEC_SIZE
    specs = []
    for i in range(num_specs):
        chunk = data[i * SPEC_SIZE:(i + 1) * SPEC_SIZE]
        specs.append(chunk.reshape(NUM_MELS, NUM_FRAMES))
    return specs

def main():
    base_dir = r"input/30_09_01/Spectrograms"
    out_pdf = "output/spectrogram_pages.pdf"

    all_specs = []
    titles = []

    # Load and collect all spectrograms (multiple per file allowed)
    for fname in sorted(os.listdir(base_dir)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(base_dir, fname)
            specs = load_spectrograms(path)
            for i, spec in enumerate(specs):
                all_specs.append(spec)
                titles.append(f"{fname}  [{i+1}]")

    if not all_specs:
        print("No spectrograms found")
        return

    # Shared color scale across all plots
    vmin = min(np.min(s) for s in all_specs)
    vmax = max(np.max(s) for s in all_specs)

    per_page = 20
    n_pages = math.ceil(len(all_specs) / per_page)

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            start = page * per_page
            end = min(start + per_page, len(all_specs))
            chunk_specs = all_specs[start:end]
            chunk_titles = titles[start:end]

            n = len(chunk_specs)
            ncols, nrows = 5, 4  # 5 columns x 4 rows = 20 per page

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(ncols * 3.6, nrows * 2.8),  # adjust to taste
                squeeze=False
            )

            im = None
            for idx, (spec, title) in enumerate(zip(chunk_specs, chunk_titles)):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                im = ax.imshow(spec, aspect="auto", origin="lower",
                               cmap="magma", vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=7)
                ax.set_xlabel("Frames", fontsize=7)
                ax.set_ylabel("Frequency (Hz)", fontsize=7)
                ax.tick_params(axis='both', which='major', labelsize=6)

            # Hide unused axes on the last page
            for idx in range(n, nrows * ncols):
                r, c = divmod(idx, ncols)
                axes[r][c].axis("off")

            # Shared colorbar
            fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.25)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
            if im is not None:
                fig.colorbar(im, cax=cbar_ax, label="Normalized dB")

            fig.suptitle(f"Spectrograms {start+1}–{end} of {len(all_specs)}", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote PDF: {out_pdf}")

if __name__ == "__main__":
    main()
