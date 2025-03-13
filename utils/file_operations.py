def read_file(file_path):
    points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                values = line.split(";")
                if len(values) == 3:
                    try:
                        points.append(tuple(map(float, values)))
                    except ValueError:
                        print(f"Línea inválida: {line}")
    return points

def save_file(file_path, points):
    with open(file_path, 'w', encoding='utf-8') as f:
        for x, y, z in points:
            f.write(f"{x};{y};{z}\n")
