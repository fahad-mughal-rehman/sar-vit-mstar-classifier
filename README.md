# SAR Vehicle Classification with Vision Transformers (MSTAR)

Fine-tuning and comparing modern Vision Transformer (ViT) backbones to classify military
vehicles from Synthetic Aperture Radar (SAR) imagery, using the MSTAR benchmark dataset.

Rather than picking one pretrained backbone on faith, this project **screens three modern ViT
variants under an identical training recipe**, promotes the winner to a full training run, and
reports a held-out test accuracy that was never touched during model selection.

## Results

**Backbone screening** (6-epoch quick run per candidate, selected on a validation split):

| Backbone | Val. accuracy | Params | Notes |
|---|---|---|---|
| **DeiT3-Small** (`deit3_small_patch16_224.fb_in22k_ft_in1k`) | **98.3%** | 21.5M | Winner |
| ViT-Small / AugReg (`vit_small_patch16_224.augreg_in21k_ft_in1k`) | 96.9% | 21.5M | Strong, standard baseline |
| EVA-02-Small (`eva02_small_patch14_224.mim_in22k`) | 55.7% | 21.5M | Underperformed at this epoch budget/LR — see notes below |

**Final model** (DeiT3-Small, trained 20 epochs, evaluated once on the held-out test set):

| Metric | Score |
|---|---|
| Test accuracy | **99.2%** |
| Macro F1 | **0.991** |

Per-class performance is near-perfect across the board; the small remaining confusion sits
between **BMP2 / BTR60 / BTR70 / T72** — visually similar tracked/wheeled armored vehicles, a
well-known hard case in the MSTAR literature, not an artifact of this pipeline.

> **Note on EVA-02:** its weak screening result almost certainly reflects a training-recipe
> mismatch (learning rate / warmup schedule tuned for a different architecture family) rather
> than the architecture being worse in general — EVA-02 is a strong, modern backbone in other
> contexts. This is exactly why the comparison is run empirically instead of assumed from a
> model's release date.

## Methodology

1. **Data**: real MSTAR SAR chips, 10 vehicle classes (`2S1, BMP2, BRDM2, BTR60, BTR70, D7, T62,
   T72, ZIL131, ZSU_23_4`). SLICY (a calibration target, not a vehicle) is excluded, matching
   standard MSTAR usage.
2. **Split**: a stratified validation set is carved out of the training data (15% per class) for
   backbone screening and checkpoint selection. The test set is touched exactly once, at the
   very end, for final reporting — avoiding the common shortcut of selecting a checkpoint on the
   same data used to report accuracy.
3. **Augmentation**: rotation, horizontal flip, and multiplicative speckle noise — the actual
   noise model for coherent radar imagery, not the additive-Gaussian assumptions built into
   typical photo-augmentation libraries.
4. **Training recipe**: AdamW, cosine learning-rate schedule with linear warmup, label smoothing,
   gradient clipping, mixed-precision (AMP) on GPU.
5. **Backbone comparison → full training**: three modern `timm` ViT variants are screened under
   identical conditions; the winner is trained to completion and is the only model evaluated on
   the test set.

## Repo structure

```
sar-vit-mstar-classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── mstar_vit_experiment.ipynb   # full pipeline: data -> screening -> training -> eval -> gallery
└── results/
    └── (generated figures go here when you run the notebook — see below)
```

## Running it yourself

1. Get the dataset: [MSTAR-10-Classes on Kaggle](https://www.kaggle.com/datasets/ravenchencn/mstar-10-classes)
   (or point `DATA_DIR` in the notebook at any MSTAR mirror with a `train/<class>/` and
   `test/<class>/` layout).
2. Open `notebooks/mstar_vit_experiment.ipynb` in Kaggle, Colab, or locally with a GPU runtime.
3. Install dependencies: `pip install -r requirements.txt`
4. Set `DATA_DIR` in the config cell to your dataset path, then run all cells top to bottom.
5. Generated figures (class counts, backbone comparison, training curves, confusion matrix,
   prediction gallery) save to `outputs/` — copy the ones you want into `results/` here.

## Dataset attribution

MSTAR (Moving and Stationary Target Acquisition and Recognition) is a public U.S. Air
Force–sponsored SAR benchmark. This project uses the
[`ravenchencn/mstar-10-classes`](https://www.kaggle.com/datasets/ravenchencn/mstar-10-classes)
Kaggle mirror. Check that dataset's page for its license/usage terms before redistributing data.

## License

Code in this repository is released under the MIT License (see `LICENSE`). This does not cover
the MSTAR dataset itself, which is not included in this repo.
