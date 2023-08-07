import numpy as np
from scipy import stats

def perform_ttest(data_A, data_B):
    t_statistic, p_value = stats.ttest_ind(data_A, data_B)

    return t_statistic, p_value

if __name__ == "__main__":
    print("Enter the conversion rates for website design A (separated by spaces):")
    design_A_data = list(map(float, input().split()))

    print("Enter the conversion rates for website design B (separated by spaces):")
    design_B_data = list(map(float, input().split()))
    
    t_statistic, p_value = perform_ttest(design_A_data, design_B_data)
    
    alpha = 0.05  

    print(f"t-statistic: {t_statistic:.3f}")
    print(f"p-value: {p_value:.3f}")

    if p_value < alpha:
        print("There is a statistically significant difference in the mean conversion rates between website design A and website design B.")
    else:
        print("There is no statistically significant difference in the mean conversion rates between website design A and website design B.")
