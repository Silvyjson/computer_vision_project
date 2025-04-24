from pycocotools.coco import COCO

def convert_label_ids_to_names(coco_annotation_file: str, predictions: list) -> list:
    """
    Convert label IDs in the predictions to label names using COCO annotations.

    Parameters:
        coco_annotation_file (str): Path to the COCO annotation JSON file.
        predictions (list): List of prediction lists per frame. Each prediction should contain 'label' (as ID).

    Returns:
        list: Updated predictions with 'label' as class name instead of ID.
    """
    coco = COCO(coco_annotation_file)
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    updated_predictions = []

    for frame_preds in predictions:
        updated_frame = []

        for pred in frame_preds:
            label_id = pred['label']
            label_name = cat_id_to_name.get(label_id, f"unknown_{label_id}")
            
            updated_pred = pred.copy()
            updated_pred['label'] = label_name

            updated_frame.append(updated_pred)

        updated_predictions.append(updated_frame)

    return updated_predictions

