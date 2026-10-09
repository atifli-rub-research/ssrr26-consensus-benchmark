def compute_observation_metrics(drone_predictions: list[dict]) -> dict:
    """Compute the existing per-drone summary across all tile predictions."""
    num_tiles = len(drone_predictions)
    total_detections = sum(prediction["num_detections"] for prediction in drone_predictions)
    tiles_with_detections = sum(prediction["num_detections"] > 0 for prediction in drone_predictions)
    total_true_mines = sum(prediction["true_mine_count"] for prediction in drone_predictions)
    tiles_with_true_mines = sum(prediction["true_mine_present"] for prediction in drone_predictions)

    confidences = [
        detection["confidence"]
        for prediction in drone_predictions
        for detection in prediction["detections"]
    ]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "num_tiles": num_tiles,
        "total_detections": total_detections,
        "tiles_with_detections": tiles_with_detections,
        "total_true_mines": total_true_mines,
        "tiles_with_true_mines": tiles_with_true_mines,
        "mean_confidence": mean_confidence,
    }
