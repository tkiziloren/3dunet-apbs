import numpy as np
from potsim2.PotGrid import PotGrid

def main():
    # PDB dosyasının yolu
    pdb_filename = "/Volumes/Data/DiskYedek/DATA/src_data/pdbbind/refined-set/1a1e/1a1e_protein.pdb"

    # PotGrid nesnesini oluşturun
    grid = PotGrid(pdb_filename=pdb_filename)

    # Protein merkezini hesaplayın
    protein_center = grid.protein_center
    print(f"Protein Center: {protein_center}")

    # Cilt maskesi oluşturun ve uygulayın
    skinMask = grid.get_skin_mask()
    grid.apply_mask(skinMask)

    # Grid verilerini dışa aktarın
    grid.export("path/to/your/output_grid.dx")

if __name__ == "__main__":
    main()