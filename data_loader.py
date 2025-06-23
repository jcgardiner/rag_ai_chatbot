def load_dataset(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file if line.strip()]
    print(f"✅ Loaded {len(data)} entries from {file_path}")
    return data