import numpy as np


def MannWhitneyU(Sample1, Sample2):
    Sample1 = np.array(Sample1)
    Sample2 = np.array(Sample2)
    NewSample = np.concatenate((Sample1, Sample2), axis=0)
    NewRanks, Groups = get_fract_ranks_and_groups(NewSample)
    SumRanks = 0
    SumRanks2 = 0
    for i in range(Sample1.shape[0]):
        SumRanks += NewRanks[i]
        SumRanks2 += NewRanks[Sample1.shape[0]+i]
    U1 = SumRanks - Sample1.shape[0]*(Sample1.shape[0]+1.0)/2.0
    U2 = SumRanks2 - Sample2.shape[0]*(Sample2.shape[0]+1.0)/2.0
    Umean = Sample1.shape[0]*Sample2.shape[0]/2.0
    GroupsSum = 0
    for index in Groups:
        GroupsSum += (index*index*index - index)/12
    N = Sample1.shape[0]+Sample2.shape[0]
    part1 = Sample1.shape[0]*Sample2.shape[0]/(N*(N-1.0))
    part2 = (N*N*N-N)/12.0
    Ucorr2 = np.sqrt(part1*(part2-GroupsSum))
    Z1 = (U1 - Umean)/Ucorr2
    Z2 = (U2 - Umean)/Ucorr2
    if(Z1 <= Z2):
        if(Z1 < -2.58):
            return "worse"
    else:
        if(Z2 < -2.58):
            return "better"
    return "equal"


def get_fract_ranks_and_groups(data):
    sort_index = np.argsort(-data)
    sort_list = -np.sort(-data)
    groups = []
    my_new_ranks = np.zeros(data.shape[0])
    counter = 0
    while(True):
        if(counter == data.shape[0]):
            break
        if(counter == data.shape[0]-1):
            my_new_ranks[counter] = counter
            break
        if(sort_list[counter] != sort_list[counter+1]):
            my_new_ranks[counter] = counter
            counter+=1            
        else:
            avgrank = 0
            start = counter
            while(sort_list[start] == sort_list[counter]):
                avgrank += counter
                counter+=1                
                if(counter == data.shape[0]):
                    break
            avgrank = avgrank / (counter - start)
            groups.append(counter - start)
            for i in range(start,counter):
                my_new_ranks[i] = avgrank
    index_rank = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        new_rank_inv = data.shape[0] - my_new_ranks[i]
        index_rank[sort_index[i]] = new_rank_inv   
    return index_rank, groups

# Sample1 = [0,0,0,0,0,0,0,0,0,0]
# Sample2 = [1,1,1,1,1,1,1,1,1,1]
# print(Sample1)
# print(Sample2)
# print(MannWhitneyU(Sample1,Sample2))
