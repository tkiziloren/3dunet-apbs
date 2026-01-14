import yaml
from pathlib import Path

# Yeni parametreler
new_training_params = {
    'accumulation_steps': 8,
}

new_model_params = {
    'dropout': 0.2
}

# Tüm yml dosyalarını güncelle
for yml_file in Path('.').glob('*.yml'):
    with open(yml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Training parametreleri ekle
    if 'training' in config:
        config['training'].update(new_training_params)
    
    # Model parametreleri ekle (yoksa oluştur)
    if 'model' not in config:
        config['model'] = {}
    config['model'].update(new_model_params)
    
    # Kaydet
    with open(yml_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Updated: {yml_file.name}")

print("\n✅ All configs updated with accumulation_steps and dropout!")
