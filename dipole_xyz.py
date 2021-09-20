import os, numpy as np
from runQOOH2D import qooh
from McUtils.Data import AtomData, UnitsData
from McUtils.Plots import Plot
# inputs I don't know
dip_dict = qooh.DipoleMomentSurface
rachel_want_to_view_keys = list(filter(lambda k: k[0] == 90, dip_dict.keys()))
qooh_atoms = qooh.atom_array
qooh_masses = [AtomData[a]["Mass"] * UnitsData.convert("AtomicMassUnits", "ElectronMass") for a in qooh_atoms]
xyz_save_dir = os.path.join(qooh.MoleculeDir, "DipoleTests")
# code to make an XYZ string from Cartesian coords
def make_a_single_xyz(atoms, carts):
    return "{}\n".format(len(atoms)) + "comment \n" + "\n".join(
           " {atom:<2} {x:>12.5f} {y:>12.5f} {z:>12.5f}".format(
             atom=a,
             x=cart[0],
             y=cart[1],
             z=cart[2]
             ) for a, cart in zip(atoms, carts)
             )
def make_an_xyz(atoms, carts_stack):
    return "\n\n".join(make_a_single_xyz(atoms, carts) for carts in carts_stack)
# pull carts out of dip dict by taking the 4th column onwards and reshaping
# don't forget that we have multiple OH vals per key
# then write these to a file
for key in rachel_want_to_view_keys:
    carts = dip_dict[key][:, 4:].reshape(len(dip_dict[key]), len(qooh_atoms), 3) / np.sqrt(qooh_masses)[np.newaxis, :, np.newaxis]
    my_xyz_string = make_an_xyz(qooh_atoms, carts)
    with open(os.path.join(xyz_save_dir, "unrot_qooh_{}_{}.xyz".format(key[0], key[1])), "w+") as output_file:
        output_file.write(my_xyz_string)
    my_xyz_ploot = Plot(dip_dict[key][:, 0], dip_dict[key][:, 1])
    Plot(dip_dict[key][:, 0], dip_dict[key][:, 2], figure=my_xyz_ploot)
    Plot(dip_dict[key][:, 0], dip_dict[key][:, 3], figure=my_xyz_ploot)
    my_xyz_ploot.savefig(os.path.join(xyz_save_dir, "unrot_qooh_{}_{}.png".format(key[0], key[1])))
