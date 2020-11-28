import numpy as np


def paired_permutation(X, Y, num_perm):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        num_perm ([type]): [description]

    Returns:
        [type]: [description]
    """    
    n = len(X)
    true_score = np.mean(X - Y)

    distribution = np.zeros((1, num_perm))

    shuffle = [ones(1,n/2) -1*ones(1,n/2)]; 
    for num in range(num_perm):
        s = shuffle(randperm(n))
        distribution(i)=mean(s.*(X-Y)); %randomly swapy xj-yj with yj-xj

    p_val = mean(true_score > distribution)

    return p_val


if __name__ == '__main__':
    num_experiments = 1000
    num_permutations = 5000
    sample_size = 30

    ps = np.zeros(1, num_experiments)

    for experiment in range(num_experiments):
        X = np.random.rand(1, sample_size)
        Y = np.random.rand(1, sample_size)
        ps[experiment] = paired_permutation(X, Y, num_permutations)

    print(np.mean(ps < 0.05))
    hist(ps)