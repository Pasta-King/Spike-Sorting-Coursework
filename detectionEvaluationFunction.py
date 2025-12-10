import numpy as np

def indexDelta50Verification(Index, idx):
    
    right = 0
    wrong = 0
    close = 0
    for i in idx:
        if i in Index:
            right += 1
        elif i + 1 in Index:
            close += 1
        elif i + 2 in Index:
            close += 1
        elif i + 3 in Index:
            close += 1
        elif i + 4 in Index:
            close += 1
        elif i + 5 in Index:
            close += 1
        elif i + 6 in Index:
            close += 1
        elif i + 7 in Index:
            close += 1
        elif i + 8 in Index:
            close += 1
        elif i + 9 in Index:
            close += 1
        elif i + 10 in Index:
            close += 1
        elif i + 11 in Index:
            close += 1
        elif i + 12 in Index:
            close += 1
        elif i + 13 in Index:
            close += 1
        elif i + 14 in Index:
            close += 1
        elif i + 15 in Index:
            close += 1
        elif i + 16 in Index:
            close += 1
        elif i + 17 in Index:
            close += 1
        elif i + 18 in Index:
            close += 1
        elif i + 19 in Index:
            close += 1
        elif i + 20 in Index:
            close += 1
        elif i + 21 in Index:
            close += 1
        elif i + 22 in Index:
            close += 1
        elif i + 23 in Index:
            close += 1
        elif i + 24 in Index:
            close += 1
        elif i + 25 in Index:
            close += 1
        elif i + 26 in Index:
            close += 1
        elif i + 27 in Index:
            close += 1
        elif i + 28 in Index:
            close += 1
        elif i + 29 in Index:
            close += 1
        elif i + 30 in Index:
            close += 1
        elif i + 31 in Index:
            close += 1
        elif i + 32 in Index:
            close += 1
        elif i + 33 in Index:
            close += 1
        elif i + 34 in Index:
            close += 1
        elif i + 35 in Index:
            close += 1
        elif i + 36 in Index:
            close += 1
        elif i + 37 in Index:
            close += 1
        elif i + 38 in Index:
            close += 1
        elif i + 39 in Index:
            close += 1
        elif i + 40 in Index:
            close += 1
        elif i + 41 in Index:
            close += 1
        elif i + 42 in Index:
            close += 1
        elif i + 43 in Index:
            close += 1
        elif i + 44 in Index:
            close += 1
        elif i + 45 in Index:
            close += 1
        elif i + 46 in Index:
            close += 1
        elif i + 47 in Index:
            close += 1
        elif i + 48 in Index:
            close += 1
        elif i + 49 in Index:
            close += 1
        elif i + 50 in Index:
            close += 1
        elif i - 1 in Index:
            close += 1
        elif i - 2 in Index:
            close += 1
        elif i - 3 in Index:
            close += 1
        elif i - 4 in Index:
            close += 1
        elif i - 5 in Index:
            close += 1
        elif i - 6 in Index:
            close += 1
        elif i - 7 in Index:
            close += 1
        elif i - 8 in Index:
            close += 1
        elif i - 9 in Index:
            close += 1
        elif i - 10 in Index:
            close += 1
        elif i - 11 in Index:
            close += 1
        elif i - 12 in Index:
            close += 1
        elif i - 13 in Index:
            close += 1
        elif i - 14 in Index:
            close += 1
        elif i - 15 in Index:
            close += 1
        elif i - 16 in Index:
            close += 1
        elif i - 17 in Index:
            close += 1
        elif i - 18 in Index:
            close += 1
        elif i - 19 in Index:
            close += 1
        elif i - 20 in Index:
            close += 1
        elif i - 21 in Index:
            close += 1
        elif i - 22 in Index:
            close += 1
        elif i - 23 in Index:
            close += 1
        elif i - 24 in Index:
            close += 1
        elif i - 25 in Index:
            close += 1
        elif i - 26 in Index:
            close += 1
        elif i - 27 in Index:
            close += 1
        elif i - 28 in Index:
            close += 1
        elif i - 29 in Index:
            close += 1
        elif i - 30 in Index:
            close += 1
        elif i - 31 in Index:
            close += 1
        elif i - 32 in Index:
            close += 1
        elif i - 33 in Index:
            close += 1
        elif i - 34 in Index:
            close += 1
        elif i - 35 in Index:
            close += 1
        elif i - 36 in Index:
            close += 1
        elif i - 37 in Index:
            close += 1
        elif i - 38 in Index:
            close += 1
        elif i - 39 in Index:
            close += 1
        elif i - 40 in Index:
            close += 1
        elif i - 41 in Index:
            close += 1
        elif i - 42 in Index:
            close += 1
        elif i - 43 in Index:
            close += 1
        elif i - 44 in Index:
            close += 1
        elif i - 45 in Index:
            close += 1
        elif i - 46 in Index:
            close += 1
        elif i - 47 in Index:
            close += 1
        elif i - 48 in Index:
            close += 1
        elif i - 49 in Index:
            close += 1
        elif i - 50 in Index:
            close += 1
        else: 
            wrong += 1
            
    print(f"{right} correctly indexed peaks, {wrong} inaccurately detected, {right+wrong+close} total peaks detected. It is {right+wrong+close==len(Index)} that the same number of peaks have been detected as the provided data. {close+right} detected within +-50 of a peak hence {100*wrong/(wrong+close+right)}% indexes wrong.")