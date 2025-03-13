def calculate_statistics(points):
    if not points:
        return {"count": 0, "max": {"x": "N/A", "y": "N/A", "z": "N/A"}, "min": {"x": "N/A", "y": "N/A", "z": "N/A"}}

    x_vals, y_vals, z_vals = zip(*points)
    return {
        "count": len(points),
        "max": {"x": max(x_vals), "y": max(y_vals), "z": max(z_vals)},
        "min": {"x": min(x_vals), "y": min(y_vals), "z": min(z_vals)},
    }
