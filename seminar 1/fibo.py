def fibo(n:int) -> int:
    f1 = 1
    f2 = 1
    i = 0
    while i < n - 2:
        f1, f2 = f1+f2, f1
        i += 1
    print(f2)

