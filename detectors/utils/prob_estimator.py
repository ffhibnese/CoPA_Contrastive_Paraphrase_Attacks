import glob
import json
import numpy as np


class ProbEstimator:
    # estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
    def __init__(self, ref_path):
        self.ref_path = ref_path
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(ref_path):
            with open(result_file, "r") as fin:
                res = json.load(fin)
                self.real_crits.extend(res["predictions"]["real"])
                self.fake_crits.extend(res["predictions"]["samples"])
        print(f"ProbEstimator: total {len(self.real_crits) * 2} samples.")

    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[
            100
        ]
        cnt_real = np.sum(
            (np.array(self.real_crits) > crit - offset)
            & (np.array(self.real_crits) < crit + offset)
        )
        cnt_fake = np.sum(
            (np.array(self.fake_crits) > crit - offset)
            & (np.array(self.fake_crits) < crit + offset)
        )
        return cnt_fake / (cnt_real + cnt_fake)
