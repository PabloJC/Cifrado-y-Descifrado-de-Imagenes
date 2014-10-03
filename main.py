from Criptosistema import encrypt
from Criptosistema import decrypt


def main():
	
	InX = [-3.0000000001,5.0000000001,-4.0000000003,2.0000000001,3.0000000001,-1.0000000001]
	a = 10; b = 28; c = (8/3); k1 = 0.05; k2 = 0.05; noise = 0

	encrypt(a,b,c,k1,k2,InX,"Lenna4.png",noise) 

	decrypt(a,b,c,k1,k2,InX,"encrypt.png")
	
main()
