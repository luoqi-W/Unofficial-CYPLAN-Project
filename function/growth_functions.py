def net_housing_change(housing_2005, housing_2025):
    """
    Calculate the net change in housing units from 2005 to 2025.
    """
    return housing_2025 - housing_2005


def net_housing_growth_rate(housing_2005, housing_2025):
    """
    Calculate the percentage change in housing stock
    relative to the initial housing stock in 2005.
    """
    if housing_2005 == 0:
        return None

    return ((housing_2025 - housing_2005) / housing_2005) * 100
  
