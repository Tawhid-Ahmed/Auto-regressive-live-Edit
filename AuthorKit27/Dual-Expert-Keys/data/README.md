# Data notes (anonymous)

This package includes **probe-augmented JSON metadata only**.

Images are **not** redistributed here (size + license). To run experiments:

1. Obtain the original **VLKEB** dataset and images from the VLKEB authors / release.
2. Place VLKEB images so that paths in the JSON resolve under:
   `data/VLKEB/VLKEB_images/mmkb_images/`
3. For EVQA stress, obtain EasyEdit multimodal / EVQA + COCO images and place them
   according to the EasyEdit layout under `data/easy-edit-mm/`.

Included JSON files:
- `data/VLKEB/grounding/grounding_train_short.json`
- `data/VLKEB/grounding/grounding_eval_short.json`
- `data/easy-edit-mm/grounding/grounding_train_short.json`
- `data/easy-edit-mm/grounding/grounding_eval_short.json`

Rebuild probes from original VLKEB/EVQA splits with:
`python tools/build_grounding_eval_json.py ...`
