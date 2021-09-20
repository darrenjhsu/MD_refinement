# antechamber_pipeline

A bash script that reads in a docked pose of a ligand and feeds this file through obabel, antechamber, and tleap, returning simulation-ready files of the respective ligand. 

## GAFF considerations: 

Atoms covered by GAFF2 are C, N, O, S, P, H, F, Cl, Br, and I. 
Any molecules containing atoms outside of this list will fail to be parameterized. 
Additionally, the molecule is expected to be closed-shell.

If the GAFF2 force field does not explicitly contain the parameters for a specific bond, angle, dihedral, etc then the parmchk2 code will fill in each missing parameter with some best estimate parameter. 
The .frcmod file created from parmchk2 will print a "penalty score" for any of these parameters; large scores should raise eyebrows. 
If any truly problematic parameters are assigned, parmchk2 will label them in the frcmod file with "ATTN, need revision"; the presence of these parameters are likely to kill/terminate any further progress for this molecule. 
See the Amber manual for further details (Amber21.pdf, pg. 304). 


## Antechamber considerations

The current implementation of the antechamber call works for neutral molecules only. 
If a charged molecule is pushed through the point charge calculation without denoting the molecular charge, then the point charges will be incorrect. 
THIS IS A BUG/IMPLEMENTATION ISSUE. 
Additionally, the antechamber run expects a complete molecule with all hydrogens present. 
Ideally, the obabel call should predict the bonding of the docked ligand and subsequently add any missing hydrogens.
THE GENERAL QUALITY OF THE OBABEL ASSIGNMENTS HAS NOT BEEN TESTED. 

A geometry optimization of the molecular structure is performed during the AM1-BCC partial charges. 
This optimization is performed in a vacuum and, so may result in poor partial charges due to reorganization of the molecular structures into a conformer that is not stable or expected in a solvent or protein environment. 
Minimized structures are saved in the sqm.pdb file, which is saved for further inspection.  

