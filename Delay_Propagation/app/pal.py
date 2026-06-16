def pali(n):
    temp = n
    reverse = 0

    while temp > 0:
        digit = temp%10
        reverse = reverse*10 + digit 
    