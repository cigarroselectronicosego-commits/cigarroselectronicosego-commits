import scipy.stats as stats
import numpy as np


def t_test(sample1, sample2, equal_var=True):
    """Perform a t-test on two samples and return the t-statistic and p-value."""
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    return t_stat, p_value


def anova_test(samples):
    """Perform ANOVA on multiple samples and return the F-statistic and p-value."""
    f_stat, p_value = stats.f_oneway(*samples)
    return f_stat, p_value


def chi_square_test(observed, expected):
    """Perform chi-square test on observed and expected frequencies."""
    chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
    return chi2_stat, p_value


def p_value_calculation(test_statistic, df):
    """Calculate the p-value from the test statistic and degrees of freedom."""
    return 1 - stats.t.cdf(test_statistic, df)


# Example usage:  
if __name__ == '__main__':
    sample1 = [2, 3, 5, 6, 8]
    sample2 = [1, 4, 5, 7, 9]
    print("T-Test:", t_test(sample1, sample2))
    
    sample3 = [1, 2, 3]
    sample4 = [4, 5, 6]
    sample5 = [7, 8, 9]
    print("ANOVA:", anova_test([sample3, sample4, sample5]))
    
    observed = [10, 20, 30]
    expected = [15, 15, 30]
    print("Chi-Square Test:", chi_square_test(observed, expected))
    
    test_statistic = 2.1 
    df = len(sample1) + len(sample2) - 2
    print("P-Value Calculation:", p_value_calculation(test_statistic, df))
