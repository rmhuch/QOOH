import numpy as np
from Converter import Constants

class TransitionMoments2D:
    """To increase flexibility, assume self.transition is list of strings relating to different transitions,
     therefore, TDM, M_coeffs, and the Dipole Moment Matrix are calculated for each transition as needed. """
    def __init__(self, OH_res, tortor_results, MolecularInfo_obj, transition=None, intensity_derivatives=False):
        self.OHresults = OH_res
        self.tortor_results = tortor_results
        self.MolecularInfo_obj = MolecularInfo_obj
        self.DipoleMomentSurface = MolecularInfo_obj.DipoleMomentSurface
        self.transition = transition
        self.intensity_derivs = intensity_derivatives
        self._InterpedGDW = None

    @property
    def InterpedGDW(self):
        if self._InterpedGDW is None:
            self._InterpedGDW = self.interpDW()
            # returns a "tuple" of grid, wfns, dips
        return self._InterpedGDW

    def easyFC(self, filename, Tstring, numGstates, numEstates):
        import csv
        Glevel = int(Tstring[0])
        Exlevel = int(Tstring[-1])
        gstorWfn = self.tortor_results[f"VOH_{Glevel}"]["wfns_array"]
        gstorEnergies = self.tortor_results[f"VOH_{Glevel}"]["energy_array"]
        extorWfn = self.tortor_results[f"VOH_{Exlevel}"]["wfns_array"]
        extorEnergies = self.tortor_results[f"VOH_{Exlevel}"]["energy_array"]
        intensities = []
        with open(filename, mode="w") as results:
            results_writer = csv.writer(results, delimiter=',')
            results_writer.writerow(["initial", "final", "Ei (cm^-1)", "frequency(cm^-1)", "FC Intensity"])
            for gState in np.arange(numGstates):
                for exState in np.arange(numEstates):
                    freq = extorEnergies[exState] - gstorEnergies[gState]
                    matEl = np.dot(gstorWfn[:, gState].T, extorWfn[:, exState])
                    intensity = abs(matEl)**2  # no units in this value
                    Ei = gstorEnergies[gState] - gstorEnergies[0]
                    intensities.append([gState, exState, freq, intensity])
                    results_writer.writerow([gState, exState, Ei, freq, intensity])
        return intensities  # ordered dict keyed by transition, holding a list of all the tor transitions

    def interpDW(self):
        """Interpolate dipole moment and wavefunction to be the same length and range"""
        from scipy import interpolate
        import matplotlib.pyplot as plt
        dat_keys = self.OHresults["data_pts"]
        wavefuns = self.OHresults["wavefunctions"]
        pot = self.OHresults["potential"]
        pot_bohr = Constants.convert(pot[:, :, 0], "angstroms", to_AU=True)  # convert to bohr
        grid_min = np.min(pot_bohr)
        grid_max = np.max(pot_bohr)
        new_grid = np.linspace(grid_min, grid_max, wavefuns.shape[1])
        # interpolate wavefunction to same size, but still renormalize (when using values) to be safe
        interp_wfns = np.zeros((wavefuns.shape[0], len(new_grid), wavefuns.shape[2]))
        interp_dips = np.zeros((wavefuns.shape[0], len(new_grid), 3))
        for i, key in enumerate(dat_keys):  # loop through 2d grid points
            dipole_dat = self.MolecularInfo_obj.DipoleMomentSurface[tuple(key)]  # watch this
            cut_ind = np.where(dipole_dat[:, 0] <= 1.4)[0]
            dat_cut = dipole_dat[cut_ind, :]
            oh_bohr = Constants.convert(dat_cut[:, 0], "angstroms", to_AU=True)
            for c in np.arange(3):  # loop through dipole components
                f = interpolate.interp1d(oh_bohr, dat_cut[:, c+1], kind="cubic",
                                         bounds_error=False, fill_value=(dat_cut[0, c+1], dat_cut[-1, c+1]))
                interp_dips[i, :, c] = f(new_grid)
            for s in np.arange(wavefuns.shape[2]):  # loop through OH wfns states
                f = interpolate.interp1d(pot_bohr[i, :], wavefuns[i, :, s], kind="cubic",
                                         bounds_error=False, fill_value=(wavefuns[i, 0, s], wavefuns[i, -1, s]))
                interp_wfns[i, :, s] = f(new_grid)
        return new_grid, interp_dips, interp_wfns  # dips in all au

    def plot_interpDW(self):
        import matplotlib.pyplot as plt
        import os
        g, dips, wfns = self.InterpedGDW
        g_ang = Constants.convert(g, "angstroms", to_AU=False)
        dat_keys = self.OHresults["data_pts"]
        colors = ["b", "g", "r"]
        for i, val in enumerate(dat_keys):
            fig = plt.figure()
            dipole_dat = self.MolecularInfo_obj.DipoleMomentSurface[tuple(val)]
            for j, c in enumerate(["A", "B", "C"]):
                # plt.plot(dipole_dat[:, 0], dipole_dat[:, j+1], "o", color="k")
                plt.plot(g_ang, dips[i, :, j], label=f"{c} - Component", color=colors[j])
            plt.legend()
            plt.title(f"{val} HOOC, OCCX Dipole")
            plt.xlabel("OH values (angstroms)")
            plt.ylabel("Dipole Moment (a.u.)")
            plt.savefig(os.path.join(
                self.MolecularInfo_obj.MoleculeDir, "figures", "2DDips", "InterpDipole_only_{}_{}.png".format(val[0], val[1])))

    def calc_TDM(self, transition_str):
        """calculates the transition moment at each 2d scan data point value. Returns the TDM at each degree.
            Normalize the mu value with the normalization of the wavefunction!!!"""
        dat_keys = np.array(self.OHresults["data_pts"])
        un_hooc = np.unique(dat_keys[:, 0])
        un_occx = np.unique(dat_keys[:, 1])
        mus = np.zeros((3, len(un_hooc), len(un_occx)))
        Glevel = int(transition_str[0])
        Exlevel = int(transition_str[-1])
        for i, hooc in enumerate(un_hooc):  # loop through 2d data points
            dat_inds = np.where(dat_keys[:, 0] == hooc)[0]
            for occx_ind, k in enumerate(dat_inds):
                gs_wfn = self.InterpedGDW[2][k, :, Glevel].T
                es_wfn = self.InterpedGDW[2][k, :, Exlevel].T
                es_wfn_t = es_wfn.reshape(-1, 1)
                for j in np.arange(3):  # loop through component dipoles
                    soup = np.diag(self.InterpedGDW[1][k, :, j]).dot(es_wfn_t)
                    mu = gs_wfn.dot(soup)
                    normMu = mu / (gs_wfn.dot(gs_wfn) * es_wfn.dot(es_wfn))  # a triple check that everything is normalized
                    mus[j, i, occx_ind] = normMu
        return mus  # in all au, returns only the TDM of the given transition (transition_str), in HOOCxOCCX

    def plot_TDM(self, filename=None, plot_cut=False):
        import matplotlib.pyplot as plt
        params = {'text.usetex': False,
                  'mathtext.fontset': 'dejavusans',
                  'font.size': 18}
        plt.rc('axes', labelsize=20)
        plt.rc('axes', labelsize=20)
        plt.rcParams.update(params)
        for Tstring in self.transition:
            TDM = self.calc_TDM(Tstring)
            fit_TDM = self.fit_TDM(TDM)
            dat_keys = np.array(self.OHresults["data_pts"])
            fit_grid = self.tortor_results["VOH_0"]["grid"][0]
            x_grid, y_grid = np.moveaxis(fit_grid, -1, 0)
            if plot_cut:
                for c, val in enumerate(["A", "B", "C"]):
                    # fig = plt.figure(figsize=(7, 7), dpi=600)
                    plt.plot(np.unique(dat_keys[:, 0]), TDM[c, :, 5])
                plt.show()
            else:
                for c, val in enumerate(["A", "B", "C"]):
                    fig = plt.figure(figsize=(7, 7), dpi=600)
                    plt.tricontourf(x_grid.flatten(), y_grid.flatten(), (fit_TDM[c, :, :] / 0.393456).flatten())
                    plt.title(f"{val}-Component")
                    plt.plot(np.repeat(113, 10), np.linspace(0, 360, 10), '--', color="k", label=r"$\tau_{eq}$")
                    plt.plot(np.repeat(261, 10), np.linspace(0, 360, 10), '--', color="k")
                    plt.xlabel("HOOC (Degrees)")
                    plt.xticks(np.arange(0, 390, 60))
                    plt.ylabel("OCCX (Degrees)")
                    plt.yticks(np.arange(0, 390, 60))
                    cbar = plt.colorbar()
                    cbar.set_label("TDM (Debye)")
                    if filename is None:
                        plt.show()
                    else:
                        file_str = f"{filename}tilefit_hoocEq_{Tstring[0]}{Tstring[-1]}_{val}.jpg"
                        fig.savefig(file_str, dpi=fig.dpi, bbox_inches="tight")
                        print(f"Figure saved to {file_str}")

    def tile_array(self, vals, grid, steps, axis):
        """
        tiles the array along the given axis
        assumes grid is shaped like a meshgrid, if it isn't use `np.moveaxis(grid, -1, 0)` to make it look like one
        """
        axis_bounds = np.unique(grid[axis].flatten())
        axis_span = axis_bounds[-1] - axis_bounds[0]
        tile_grids = []
        val_grids = []
        for n in range(-steps, steps + 1):
            if n == 0:
                tile_grids.append(grid)
                val_grids.append(vals)
                continue
            subgrid = grid.copy()
            new_axis_chunk = subgrid[axis] + n * axis_span
            subgrid[axis] = new_axis_chunk
            # we need to drop one point from both the chunk
            # and the vals to avoid dupes
            sel_spec = (
                    (slice(None, None),) * axis +
                    ((slice(1, None),) if n > 0 else (slice(None, -1),))
                    + (slice(None, None),) * (len(grid) - axis - 1)
            )
            subgrid = [g[sel_spec] for g in subgrid]
            subvals = vals[sel_spec]
            tile_grids.append(subgrid)
            val_grids.append(subvals)
        new_grid = [np.concatenate([x[i] for x in tile_grids], axis=axis) for i in range(len(grid))]
        new_vals = np.concatenate(val_grids, axis=axis)
        return new_vals, new_grid

    def fit_TDM(self, TDM):
        """This will tile the TDM so that we can "force" a periodic fit"""
        from scipy import interpolate
        wfn_grid = self.tortor_results["VOH_0"]["grid"][0]
        xnew = np.unique(wfn_grid[:, 0])
        ynew = np.unique(wfn_grid[:, 1])
        currentTDMgrid = np.array(self.OHresults["data_pts"])
        TDMgrid_reshape = currentTDMgrid.reshape((13, 13, 2))
        grid_for_real = np.moveaxis(TDMgrid_reshape, -1, 0)
        fit_TDM = np.zeros((3, len(xnew), len(ynew)))
        for i in np.arange(3):  # loop through components
            tile1_vals, tile1_grid = self.tile_array(TDM[i], grid_for_real, steps=1, axis=0)
            tile2_vals, tile2_grid = self.tile_array(tile1_vals, tile1_grid, steps=1, axis=1)
            f = interpolate.interp2d(np.unique(tile2_grid[0].flatten()), np.unique(tile2_grid[1].flatten()), tile2_vals,
                                     kind="cubic")
            # Transpose tdm because interp2d expects (y,x) matrix
            fit_TDM[i] = f(xnew, ynew)
        return fit_TDM

    def calc_TDM_intensity(self, filename, Tstring, numGstates, numEstates):
        import csv
        Glevel = int(Tstring[0])
        Exlevel = int(Tstring[-1])
        gstorWfn = self.tortor_results[f"VOH_{Glevel}"]["wfns_array"]
        gstorEnergies = self.tortor_results[f"VOH_{Glevel}"]["energy_array"]
        extorWfn = self.tortor_results[f"VOH_{Exlevel}"]["wfns_array"]
        extorEnergies = self.tortor_results[f"VOH_{Exlevel}"]["energy_array"]
        intensities = []
        TDM = self.calc_TDM(Tstring)
        fit_TDM = self.fit_TDM(TDM)
        with open(filename, mode="w") as results:
            results_writer = csv.writer(results, delimiter=',')
            results_writer.writerow(["initial", "final", "Ei (cm^-1)", "frequency(cm^-1)", "mu_a(Debye)",
                                         "mu_b", "mu_c", "tot_intensity(km/mol)"])
            for gState in np.arange(numGstates):
                for exState in np.arange(numEstates):
                    freq = extorEnergies[exState] - gstorEnergies[gState]  # already in wavenumbers
                    matEl = np.zeros(3)
                    matEl_D = np.zeros(3)
                    comp_intents = np.zeros(3)
                    for c, val in enumerate(["A", "B", "C"]):  # loop through components
                        matEl[c] = (np.dot(gstorWfn[:, gState], (fit_TDM[c, :, :].flatten() * extorWfn[:, exState])))**2
                        matEl_D[c] = matEl[c] / 0.393456
                        comp_intents[c] = matEl[c] * freq * 2.506 / (0.393456 ** 2)
                    intensity = np.sum(comp_intents)
                    Ei = gstorEnergies[gState]-gstorEnergies[0]
                    intensities.append([gState, exState, freq, intensity])
                    results_writer.writerow([gState, exState, Ei, freq, *matEl_D, intensity])
        return intensities  # ordered dict keyed by transition, holding a list of all the tor transitions

    # calculate overlap of torsion wfns with TDM
    def calc_intensity(self, numGstates=4, numEstates=4, FC=True):
        from collections import OrderedDict
        all_intensities = OrderedDict()
        for Tstring in self.transition:  # should be a list of strings
            print(Tstring)
            Glevel = int(Tstring[0])
            Exlevel = int(Tstring[-1])
            if FC:
                filename = f"TransitionIntensities_vOH{Glevel}tovOH{Exlevel}_2DFC.csv"
                print(f"2D FC Energies/Wavefunctions saved to {filename}")
                all_intensities[Tstring] = self.easyFC(filename, Tstring, numGstates, numEstates)
            else:
                filename = f"TransitionIntensities_vOH{Glevel}tovOH{Exlevel}_2DTDM_symmTDMtestAA.csv"
                print(f"2D TDM Energies/Wavefunctions saved to {filename}")
                all_intensities[Tstring] = self.calc_TDM_intensity(filename, Tstring, numGstates, numEstates)
        return all_intensities

    def plot_sticks(self, numGstates=4, numEstates=4, FC=False):
        import matplotlib.pyplot as plt
        params = {'text.usetex': False,
                  'mathtext.fontset': 'dejavusans',
                  'font.size': 10}
        plt.rcParams.update(params)
        dat = self.calc_intensity(numGstates=numGstates, numEstates=numEstates, FC=FC)
        if FC:
            file_tag = "FC"
        else:
            file_tag = "TDM"
        # lims = [(3600, 3900), (7000, 7250), (10250, 10550), (13300, 13900), (16300, 16600)]
        for i, Tstring in enumerate(self.transition):
            spect_dat = np.array(dat[Tstring])
            fig = plt.figure(figsize=(8, 5), dpi=600)
            mkline, stline, baseline = plt.stem(spect_dat[:, 2], spect_dat[:, 5]/spect_dat[0, 5], linefmt="-C1",
                                                markerfmt=' ', basefmt="-k", use_line_collection=True)
            plt.setp(stline, "linewidth", 1.5)
            plt.setp(baseline, "linewidth", 0.5)
            plt.yticks(visible=False)
            # plt.xlim(*lims[i])
            plt.savefig(f"stickspect_{file_tag}_{Tstring[0]}{Tstring[-1]}.jpg", dpi=fig.dpi, bbox_inches="tight")

