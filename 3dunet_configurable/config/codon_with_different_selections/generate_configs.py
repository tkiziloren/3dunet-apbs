#!/usr/bin/env python3
"""
Generate configuration files for different feature combinations
For each combination, generates TWO configs:
  - one with binding_site_in_dataset label
  - one with binding_site_calculated label
"""

import yaml
from pathlib import Path

# Paths
BASE_CONFIG = Path(__file__).parent.parent / "local" / "config_with_feature_selection.yml"
COMBINATIONS_FILE = Path(__file__).parent / "feature_combinations.yml"
OUTPUT_DIR = Path(__file__).parent

# Codon specific h5_directory
CODON_H5_DIR = "/hps/nobackup/arl/chembl/tevfik/deep-apbs-data/pdbbind/refined-set_filter"

# Labels to generate for each combination
LABELS = [
    "binding_site_in_dataset",
    "binding_site_calculated"
]

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, filepath):
    import re
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)
    
    # Fix list indentation - add 2 spaces before list items that start at column 0
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines):
        # If line starts with "- " and previous line ends with ":", indent it
        if line.startswith('- ') and i > 0 and lines[i-1].rstrip().endswith(':'):
            fixed_lines.append('  ' + line)
        # If line starts with "- " and previous line also starts with "  -", keep same indent
        elif line.startswith('- ') and i > 0 and fixed_lines[-1].startswith('  - '):
            fixed_lines.append('  ' + line)
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(fixed_lines)

def generate_configs():
    # Load base configuration
    base_config = load_yaml(BASE_CONFIG)
    
    # Load feature combinations
    combinations = load_yaml(COMBINATIONS_FILE)
    
    # Update h5_directory to Codon path
    base_config['h5_directory'] = CODON_H5_DIR
    
    total_configs = 0
    
    # Generate config for each combination and each label
    for name, combo in combinations['combinations'].items():
        for label in LABELS:
            # Create new config based on base
            new_config = base_config.copy()
            
            # Update features and label
            new_config['features'] = combo['features']
            new_config['label'] = label
            
            # Generate filename with label suffix
            if label == "binding_site_in_dataset":
                label_suffix = "label_dataset_binding_site"
            else:  # binding_site_calculated
                label_suffix = "label_calculated_binding_site"
            output_file = OUTPUT_DIR / f"{name}_{label_suffix}.yml"
            
            # Save to file
            save_yaml(new_config, output_file)
            print(f"✓ {output_file.name}")
            print(f"  Features: {', '.join(combo['features'][:3])}{'...' if len(combo['features']) > 3 else ''}")
            print(f"  Label: {label}")
            print()
            
            total_configs += 1
    
    return total_configs

if __name__ == "__main__":
    print("Generating configuration files for Codon cluster...")
    print("=" * 70)
    total = generate_configs()
    print("=" * 70)
    print(f"Done! Generated {total} configuration files.")
    print(f"  - {total//2} feature combinations")
    print(f"  - 2 label variants each (in_dataset, calculated)")
