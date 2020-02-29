def exo1(N):

    n=2

    L = [i for i in range(2,N)]

    all_verif = False

    while all_verif == False:
        all_verif = True
        for k in L:
            if n%k != 1:
                all_verif = False
                break
        if n%N != 0:
            all_verif = False
        n = n + 1
    return n-1

print(exo1(17))