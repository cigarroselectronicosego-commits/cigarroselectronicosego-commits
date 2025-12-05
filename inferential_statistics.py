import numpy as np
import scipy.stats as stats

class StatisticalAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def confidence_interval(data, confidence=0.95):
        mean = np.mean(data)
        sem = stats.sem(data)
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
        return mean - margin_of_error, mean + margin_of_error

    @staticmethod
    def effect_size(group1, group2):
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
        return (mean1 - mean2) / pooled_std

    @staticmethod
    def power_analysis(effect_size, n, alpha=0.05):
        power = 1 - stats.norm.cdf(stats.norm.ppf(1-alpha/2) - effect_size/np.sqrt(n))
        return power

    @staticmethod
    def statistical_inference(data, null_hypothesis_mean, alpha=0.05):
        t_stat, p_value = stats.ttest_1samp(data, null_hypothesis_mean)
        return t_stat, p_value
