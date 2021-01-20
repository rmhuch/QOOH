import numpy as np

class Intensities:
    @classmethod
    def FranckCondon(cls, gs_wfn, es_wfn):
        intensity = np.dot(gs_wfn.T, es_wfn) ** 2
        return intensity

    @classmethod
    def TDM(cls, gs_wfn, es_wfn, tdm):
        comp_intents = np.zeros(3)
        for j in np.arange(3):  # transition moment component
            super_es = tdm[j, :, :].T * es_wfn
            comp_intents[j] = np.dot(gs_wfn.T, super_es.T)
        return comp_intents
