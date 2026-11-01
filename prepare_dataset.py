import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    root_dir = "WikiArt/wikiart/versions/1"
    
    # 1. Pravidla pro sloučení příbuzných tříd
    class_mapping = {
        "Analytical_Cubism": "Cubism",
        "Synthetic_Cubism": "Cubism",
        "Early_Renaissance": "Renaissance",
        "High_Renaissance": "Renaissance",
        "Northern_Renaissance": "Renaissance",
        "Mannerism_Late_Renaissance": "Renaissance",
        "Contemporary_Realism": "Realism",
        "New_Realism": "Realism",
        # Třídy, které neměníme, se namapují na svůj původní název
    }

    data = []
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    print("Načítám cesty k obrázkům...")
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        # Použití mapování, pokud existuje, jinak ponechání původního názvu
        mapped_cls = class_mapping.get(cls_name, cls_name)
        
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Ukládáme relativní cestu
                img_path = os.path.join(cls_dir, img_name)
                data.append({"image_path": img_path, "label": mapped_cls})

    df = pd.DataFrame(data)
    print(f"Celkový počet obrázků před filtrováním: {len(df)}")

    # 2. Filtrování "ocasních" tříd (méně než 1500 obrázků)
    min_samples = 1500
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    
    df_filtered = df[df['label'].isin(valid_classes)].copy()
    print(f"Počet obrázků po odstranění tříd s méně než {min_samples} vzorky: {len(df_filtered)}")
    print(f"Výsledný počet sjednocených tříd: {len(valid_classes)}")

    # 3. Stratifikované rozdělení na Train (80 %), Val (10 %) a Test (10 %)
    # Nejprve oddělíme 80 % pro trénink a 20 % pro zbytek
    train_df, temp_df = train_test_split(
        df_filtered, 
        test_size=0.20, 
        stratify=df_filtered['label'], 
        random_state=42
    )
    
    # Zbytek (20 %) rozdělíme napůl pro validaci a testování
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['label'], 
        random_state=42
    )

    # Označení rozdělení v DataFrame
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Spojení a uložení do CSV
    final_df = pd.concat([train_df, val_df, test_df])
    final_df.to_csv("dataset_split.csv", index=False)
    
    print("\nRozdělení dokončeno a uloženo do dataset_split.csv")
    print("Statistika rozdělení:")
    print(final_df['split'].value_counts())

if __name__ == "__main__":
    main()