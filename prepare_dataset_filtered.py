import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    root_dir = "WikiArt/wikiart/versions/1"
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    data = []
    print("Načítám cesty k obrázkům...")
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({"image_path": os.path.join(cls_dir, img_name), "label": cls_name})

    df = pd.DataFrame(data)
    
    # Filtrování tříd s méně než 1500 vzorky (bez slučování)
    min_samples = 1500
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    
    df_filtered = df[df['label'].isin(valid_classes)].copy()
    print(f"Celkový počet obrázků před filtrováním: {len(df)}")
    print(f"Počet obrázků po odstranění tříd < 1500: {len(df_filtered)}")
    print(f"Výsledný počet zachovaných tříd: {len(valid_classes)}")

    # Stratifikované rozdělení 80/10/10
    train_df, temp_df = train_test_split(df_filtered, test_size=0.20, stratify=df_filtered['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv("dataset_filtered_split.csv", index=False)
    print("Rozdělení uloženo do dataset_filtered_split.csv")

if __name__ == "__main__":
    main()