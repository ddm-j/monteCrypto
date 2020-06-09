import pandas as pd
import numpy as np

def adaptiveLaguerre(data, length=10, minperiod=60):
    av = data.values#data.close.values#((data.high + data.low) / 2).values
    filt = []
    L0 = []
    L1 = []
    L2 = []
    L3 = []
    for i in range(0, len(data)):
        # Initialize Variables
        if i < minperiod:
            filt.append(av[i-1])
            L0.append(av[i])
            L1.append(av[i-1])
            L2.append(av[i-2])
            L3.append(av[i-2])
            # print(data.L0.iloc[i],data.L1.iloc[i],data.L2.iloc[i],data.L3.iloc[i])

            continue

        diff = abs(np.array(av[i-length:i])-np.array(filt[-length:]))
        HH = diff[0]
        LL = diff[0]
        for j in range(length):
            if diff[j] > HH:
                HH = diff[j]
            if diff[j] < LL:
                LL = diff[j]
        x = [(j - LL) / (HH - LL) for j in diff]
        if HH - LL != 0.0:
            a = np.median(x)

        L0.append(a * av[i-1] + (1 - a) * L0[-1])
        L1.append(-(1 - a) * L0[-1] + L0[-2] + (1 - a) * L1[-1])
        L2.append(-(1 - a) * L1[-1] + L1[-2] + (1 - a) * L2[-1])
        L3.append(-(1 - a) * L2[-1] + L2[-2] + (1 - a) * L3[-1])
        filt.append((L0[-1] + 2 * L1[-1] + 2 * L2[-1] + L3[-1]) / 6)

    filt[:minperiod] = [np.nan]*minperiod

    return data - filt

def laguerre(data, length=10, minperiod=60):
    av = ((data.high + data.low) / 2).values
    filt = []
    L0 = []
    L1 = []
    L2 = []
    L3 = []
    a = 2/(length+1)
    for i in range(0, len(data)):
        # Initialize Variables
        if i < minperiod:
            filt.append(av[i-1])
            L0.append(av[i])
            L1.append(av[i-1])
            L2.append(av[i-2])
            L3.append(av[i-2])
            # print(data.L0.iloc[i],data.L1.iloc[i],data.L2.iloc[i],data.L3.iloc[i])

            continue

        L0.append(a * av[i-1] + (1 - a) * L0[-1])
        L1.append(-(1 - a) * L0[-1] + L0[-2] + (1 - a) * L1[-1])
        L2.append(-(1 - a) * L1[-1] + L1[-2] + (1 - a) * L2[-1])
        L3.append(-(1 - a) * L2[-1] + L2[-2] + (1 - a) * L3[-1])
        filt.append((L0[-1] + 2 * L1[-1] + 2 * L2[-1] + L3[-1]) / 6)

    filt[:minperiod] = [np.nan]*minperiod
    data['filt'] = filt
    return data