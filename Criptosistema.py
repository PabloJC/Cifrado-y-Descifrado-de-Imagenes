import math
from PIL import Image
from random import randint
import Image
import numpy as np
from random import random
from random import choice

'''Substitution process'''
def systemX (a,b,c,k1,k2,InX):
	OutX = [0,0,0,0,0,0]
	OutX[0] = (a * (InX[1] - InX[0])) % (256)
	OutX[1] = (b * InX[0] - InX[1] - InX[0] * InX[2] + k1 * (InX[3] - InX[4])) % (256)
	OutX[2] = (InX[0] * InX[1] - c * InX[2]) % (256)
	OutX[3] = (a * (InX[4] - InX[3])) % (256)
	OutX[4] = (b * InX[3] - InX[4] - InX[3] * InX[5] + k2 * (InX[0] - InX[1])) % (256)
	OutX[5] = (InX[3] * InX[4] - c * InX[5]) % (256)
	return OutX
	
def generate_l_sequences(a,b,c,k1,k2,InX,l,M,N):
	
	Out_Seq = [[InX[0]],[InX[1]],[InX[2]],[InX[3]],[InX[4]],[InX[5]]]
	seq_In = InX
	sequence_size = 2
	
	while sequence_size <= (l + 2 * M * N):
		
		seq_Out = systemX(a,b,c,k1,k2,seq_In)
		seq_In = seq_Out
		seq1 = Out_Seq[0]
		seq1.append(seq_Out[0])
		seq2 = Out_Seq[1]
		seq2.append(seq_Out[1])
		seq3 = Out_Seq[2]
		seq3.append(seq_Out[2])
		seq4 = Out_Seq[3]
		seq4.append(seq_Out[3])
		seq5 = Out_Seq[4]
		seq5.append(seq_Out[4])
		seq6 = Out_Seq[5]
		seq6.append(seq_Out[5])
		Out_Seq	= [seq1,seq2,seq3,seq4,seq5,seq6]
		sequence_size = sequence_size + 1
	return Out_Seq

def discard_old_l_values_of_sequences (Seq_X,l):
	Seq_1 = Seq_X[0]
	Seq_2 = Seq_X[1]
	Seq_3 = Seq_X[2]
	Seq_4 = Seq_X[3]
	Seq_5 = Seq_X[4]
	Seq_6 = Seq_X[5]
	
	Seq_1 = Seq_1[l:]
	Seq_2 = Seq_2[l:]
	Seq_3 = Seq_3[l:]
	Seq_4 = Seq_4[l:]
	Seq_5 = Seq_5[l:]
	Seq_6 = Seq_6[l:]
	
	return [Seq_1,Seq_2,Seq_3,Seq_4,Seq_5,Seq_6]
	
def generate_Li_sequences (Seq_X,M,N,t1,t2,t3,t4,t5,t6):
	
	L1_1 = [Seq_X[0][t1]]
	L1_2 = [Seq_X[4][t3]]
	L1_3 = [Seq_X[2][t5]]
	
	L2_1 = [Seq_X[4][1]]
	L2_2 = [Seq_X[2][7]]
	L2_3 = [Seq_X[0][10]]
	
	for i in range(1,int(math.floor((M*N)/2))):
		L1_1.append(Seq_X[0][i + t1])
		L1_2.append(Seq_X[4][i + t3])
		L1_3.append(Seq_X[2][i + t5])
	for i in range(int(math.ceil((M*N)/2))):
		L1_1.append(Seq_X[3][i  + t2])
		L1_2.append(Seq_X[1][i  + t4])
		L1_3.append(Seq_X[5][i  + t6])
	cont = 1
	for i in range(1,M*N):

		if (i % 2) == 0:
			L2_1.append(Seq_X[4][i + 1])
			L2_2.append(Seq_X[2][i + 7])
			L2_3.append(Seq_X[0][i + 10])
		else:
			
			L2_1.append(Seq_X[1][i + 3])
			L2_2.append(Seq_X[5][i - cont])
			L2_3.append(Seq_X[3][i - 1])
			cont +=1
		
		
	return [L1_1,L1_2,L1_3,L2_1,L2_2,L2_3]
	
def generate_ELi_sequences (Seq_Li,M,N):

	El1_1 = [10*10*(Seq_Li[0][0]) - math.ceil(10*10*(Seq_Li[0][0]))]
	El1_2 = [10*10*(Seq_Li[1][0]) - math.ceil(10*10*(Seq_Li[1][0]))]
	El1_3 = [10*10*(Seq_Li[2][0]) - math.ceil(10*10*(Seq_Li[2][0]))]
	El2_1 = [10*10*(Seq_Li[3][0]) - math.ceil(10*10*(Seq_Li[3][0]))]
	El2_2 = [10*10*(Seq_Li[4][0]) - math.ceil(10*10*(Seq_Li[4][0]))]
	El2_3 = [10*10*(Seq_Li[5][0]) - math.ceil(10*10*(Seq_Li[5][0]))]
	for i in range(1,M*N):
		El1_1.append(10*10*(Seq_Li[0][i]) - math.ceil(10*10*(Seq_Li[0][i])))
		El1_2.append(10*10*(Seq_Li[1][i]) - math.ceil(10*10*(Seq_Li[1][i])))
		El1_3.append(10*10*(Seq_Li[2][i]) - math.ceil(10*10*(Seq_Li[2][i])))
		El2_1.append(10*10*(Seq_Li[3][i]) - math.ceil(10*10*(Seq_Li[3][i])))
		El2_2.append(10*10*(Seq_Li[4][i]) - math.ceil(10*10*(Seq_Li[4][i])))
		El2_3.append(10*10*(Seq_Li[5][i]) - math.ceil(10*10*(Seq_Li[5][i])))
	return [El1_1,El1_2,El1_3,El2_1,El2_2,El2_3]
	
def sorted_Sequences_and_index(Seq_Eli,M,N):

	SEl1_1 = sorted(Seq_Eli[0],reverse = True)
	SEl1_2 = sorted(Seq_Eli[1],reverse = True)
	SEl1_3 = sorted(Seq_Eli[2],reverse = True)
	SEl2_1 = sorted(Seq_Eli[3])
	SEl2_2 = sorted(Seq_Eli[4])
	SEl2_3 = sorted(Seq_Eli[5])
	
	IEl1_1 = []
	IEl1_2 = []
	IEl1_3 = []
	IEl2_1 = []
	IEl2_2 = []
	IEl2_3 = []
	for i in range(M*N):
		IEl1_1.append(Seq_Eli[0].index(SEl1_1[i]))
		IEl1_2.append(Seq_Eli[1].index(SEl1_2[i]))
		IEl1_3.append(Seq_Eli[2].index(SEl1_3[i]))
		IEl2_1.append(Seq_Eli[3].index(SEl2_1[i]))
		IEl2_2.append(Seq_Eli[4].index(SEl2_2[i]))
		IEl2_3.append(Seq_Eli[5].index(SEl2_3[i]))
	
	
	return [SEl1_1,SEl1_2,SEl1_3,SEl2_1,SEl2_2,SEl2_3],[IEl1_1,IEl1_2,IEl1_3,IEl2_1,IEl2_2,IEl2_3]
	

def quantify_index_sequences (Seq_SSI,M,N):

	for i in range(6):
		for j in range(M*N):
			Seq_SSI[1][i][j] = (Seq_SSI[1][i][j]) % 256
			
	return Seq_SSI

def generate_diffusion_matrix(Seq,M,N):
	
	EM1_1 = np.array(Seq[1][0],int)
	EM1_1 = EM1_1.reshape((N,M))
	EM1_2 = np.array(Seq[1][1],int)
	EM1_2 = EM1_2.reshape((N,M))
	EM1_3 = np.array(Seq[1][2],int)
	EM1_3 = EM1_3.reshape((N,M))
	EM2_1 = np.array(Seq[1][3],int)
	EM2_1 = EM2_1.reshape((N,M))
	EM2_2 = np.array(Seq[1][4],int)
	EM2_2 = EM2_2.reshape((N,M))
	EM2_3 = np.array(Seq[1][5],int)
	EM2_3 = EM2_3.reshape((N,M))

	return [EM1_1,EM1_2,EM1_3,EM2_1,EM2_2,EM2_3]
	
def modify_pixel_value(R,G,B,EMi,M,N):

	for j in range(N):
		for i in range(M):
			if j == 0 :
				R_old = R.getpixel((i,0))
				R_aux = (R_old ^ EMi[0][0][i]) ^ EMi[3][0][i]
				R.putpixel((i,0),R_aux) 
				G_old = G.getpixel((i,0))
				G_aux = (G_old ^ EMi[1][0][i]) ^ EMi[4][0][i]
				G.putpixel((i,0),G_aux) 
				B_old = B.getpixel((i,0))
				B_aux = (B_old ^ EMi[2][0][i]) ^ EMi[5][0][i]
				B.putpixel((i,0),B_aux) 
			else:
				R_old = R.getpixel((i,j))
				R_aux = ((R_old ^ EMi[0][j][i]) ^ EMi[3][j][i]) ^ R.getpixel((i,j-1))
				R.putpixel((i,j),R_aux) 
				G_old = G.getpixel((i,j))
				G_aux = ((G_old ^ EMi[1][j][i]) ^ EMi[4][j][i]) ^ G.getpixel((i,j-1))
				G.putpixel((i,j),G_aux) 
				B_old = B.getpixel((i,j))
				B_aux = ((B_old ^ EMi[2][j][i]) ^ EMi[5][j][i]) ^ B.getpixel((i,j-1))
				B.putpixel((i,j),B_aux) 
				
	return R,G,B
	
'''Permutation process'''

def generate_LLi_sequences (Seq_X,M,N,l1,l2,l3,l4,l5,l6):
	
	LL1_1 = [Seq_X[5][1]]
	LL1_2 = [Seq_X[3][7]]
	LL1_3 = [Seq_X[1][10]]
	
	LL2_1 = [Seq_X[3][l1]]
	LL2_2 = [Seq_X[5][l3]]
	LL2_3 = [Seq_X[1][l5]]
	
	for i in range(1,int(math.floor((M*N)/2))):
		LL2_1.append(Seq_X[3][i + l1])
		LL2_2.append(Seq_X[5][i + l3])
		LL2_3.append(Seq_X[1][i + l5])
	for i in range(int(math.ceil((M*N)/2))):
		LL2_1.append(Seq_X[0][i  + l2])
		LL2_2.append(Seq_X[2][i  + l4])
		LL2_3.append(Seq_X[4][i  + l6])
	cont = 1
	for i in range(1,M*N):

		if (i % 2) == 0:
			LL1_1.append(Seq_X[5][i + 1])
			LL1_2.append(Seq_X[3][i + 7])
			LL1_3.append(Seq_X[1][i + 10])
		
		else:
		
			LL1_1.append(Seq_X[2][i + 3])	
			LL1_2.append(Seq_X[0][i - cont])
			LL1_3.append(Seq_X[4][i -1])
			
			cont +=1
		
	return [LL1_1,LL1_2,LL1_3,LL2_1,LL2_2,LL2_3]
	
def generate_SLi_sequences (Seq_Li,M,N):

	Sl1_1 = [10*10*(Seq_Li[0][0]) - math.ceil(10*10*(Seq_Li[0][0]))]
	Sl1_2 = [10*10*(Seq_Li[1][0]) - math.ceil(10*10*(Seq_Li[1][0]))]
	Sl1_3 = [10*10*(Seq_Li[2][0]) - math.ceil(10*10*(Seq_Li[2][0]))]
	Sl2_1 = [10*10*(Seq_Li[3][0]) - math.ceil(10*10*(Seq_Li[3][0]))]
	Sl2_2 = [10*10*(Seq_Li[4][0]) - math.ceil(10*10*(Seq_Li[4][0]))]
	Sl2_3 = [10*10*(Seq_Li[5][0]) - math.ceil(10*10*(Seq_Li[5][0]))]
	for i in range(1,M*N):
		Sl1_1.append(10*10*(Seq_Li[0][i]) - math.ceil(10*10*(Seq_Li[0][i])))
		Sl1_2.append(10*10*(Seq_Li[1][i]) - math.ceil(10*10*(Seq_Li[1][i])))
		Sl1_3.append(10*10*(Seq_Li[2][i]) - math.ceil(10*10*(Seq_Li[2][i])))
		Sl2_1.append(10*10*(Seq_Li[3][i]) - math.ceil(10*10*(Seq_Li[3][i])))
		Sl2_2.append(10*10*(Seq_Li[4][i]) - math.ceil(10*10*(Seq_Li[4][i])))
		Sl2_3.append(10*10*(Seq_Li[5][i]) - math.ceil(10*10*(Seq_Li[5][i])))
	return [Sl1_1,Sl1_2,Sl1_3,Sl2_1,Sl2_2,Sl2_3]
	
def generate_LS1_LS2_sequences(Seq_SL):

	LS1 = Seq_SL[0] + Seq_SL[1] + Seq_SL[2]
	LS2 = Seq_SL[3] + Seq_SL[4] + Seq_SL[5]
	
	return LS1,LS2
	
def div_LS1_LS3_sequences(Seq_LS,M,N):

	LS1i = []
	LS2i = []
	
	cont = 0
	for i in range(M):
		LS1i.append(Seq_LS[0][cont: (cont + 3*N)])
		cont = cont + 3*N
	cont = 0
	for i in range(N):
		LS2i.append(Seq_LS[0][cont: (cont + 3*M)])
		cont = cont + 3*M
	return LS1i , LS2i
	
def sorted_LSI_Sequences_and_index(LSi_seq,M,N):
	
	SLS1 = [sorted(LSi_seq[0][0])]
	for i in range(1,len(LSi_seq[0])):
		SLS1.append(sorted(LSi_seq[0][i]))
	
	SLS2 = [sorted(LSi_seq[1][0],reverse = True)]
	for i in range(1,len(LSi_seq[1])):
		SLS2.append(sorted(LSi_seq[1][i]))

	ILS1 = []
	ILS2 = []
	
	for i in range(M):
		ILSi =[]
		for p in range(3*N):
			#ILSi =[]
			for j in range(3*N):
				
				if SLS1[i][p] == LSi_seq[0][i][j]:
					ILSi.append(j)
					#ILS0.append(j)
					break
		ILS1.append(ILSi)
		
	for i in range(N):
		ILSi =[]
		for p in range(3*M):
			#ILSi =[]
			for j in range(3*M):
				
				if SLS2[i][p] == LSi_seq[1][i][j]:
					ILSi.append(j)
					#ILS0.append(j)
					break
		ILS2.append(ILSi)
		
	return [SLS1,ILS1],[SLS2,ILS2] 
	
def generate_scrambling_matrix (Seq_LS,M,N):

	SM1 = np.array(Seq_LS[0][1],int)
	
	aux = np.array(Seq_LS[1][1],int)
	SM2 = np.transpose(aux)
	
	return SM1, SM2
	
def combinate_matrix_RGB (R,G,B,M,N):

	P1 = np.concatenate([R,G,B], axis=1)
	
	return P1
	
def shuffle_positions (P1,SM1,M,N):

	a = []
	for i in range(M):
		for j in range(3*N):
			a.append(P1[i][(SM1[i][j])])
	P = np.array(a,int)
	P = P.reshape((M,3*N))
	return P
	
	
def obtain_components_I(P):

	R,G,B = np.hsplit(P,3)

	return R,G,B
	
def combinate_matrix_col_RGB (R,G,B):

	P2 = np.concatenate((R,G,B))
	
	return P2
	
def scramble_positions (P2,SM2,M,N):

	a = []
	for i in range(3*M):
		for j in range(N):
			a.append(P2[(SM2[i][j])][j])
	P = np.array(a,int)
	P = P.reshape((3*M,N))
	return P
	
def obtain_components_CI(P):

	R,G,B = np.split(P,3)

	return R,G,B
	
def transform_image_to_value (img,M,N):

	M_aux = []
	for i in range(M):
		for j in range(N):
			M_aux.append(img.getpixel((i,j)))
	MT = np.array(M_aux,int)
	MT = MT.reshape((M,N))
	return MT

def transform_value_to_image(img_old,val,M,N):
	
	new_img = img_old
	for j in range(N):
		for i in range(M):
			new_img.putpixel((i,j),val[i][j]) 
	return new_img
			
def efectoSaltPepper(original, densidadRuido):
    width, height = original.size
    original = original.convert('L')
    modificado = Image.new(mode='L', size =(width,height))
    org = original.load()
    mod = modificado.load()
    for y in xrange(height):
        for x in xrange(width):
            pixel = org[x,y]
            if(random() <= densidadRuido):
                if(choice([True, False])):
                       mod[x, y] = 0
                else:
                       mod[x, y] = 255
            else: 
                mod[x,y] = pixel
    data = np.array(modificado)
    im = Image.fromarray(data)
    return im

def encrypt (a,b,c,k1,k2,InX,ruta,densidadRuido):
	
	im = Image.open(ruta)
	im.load()
	R,G,B = im.split()
	M,N = im.size
	l = 1
	t1,t2,t3,t4,t5,t6 = [1,2,3,4,5,6]
	l1,l2,l3,l4,l5,l6 = [7,8,9,10,11,12]
		
	chaotic_sequences = generate_l_sequences(a,b,c,k1,k2,InX,l,M,N)
	Salida2 = discard_old_l_values_of_sequences (chaotic_sequences,l)
	Salida3 = substitution(Salida2,M,N,t1,t2,t3,t4,t5,t6)
	R_new,G_new,B_new = modify_pixel_value(R,G,B,Salida3,M,N)
	
	R_aux = transform_image_to_value(R_new,M,N)
	G_aux = transform_image_to_value(G_new,M,N)
	B_aux = transform_image_to_value(B_new,M,N)
	P1 = combinate_matrix_RGB (R_aux,G_aux,B_aux,M,N)
	image_encrypt = permutation(Salida2,M,N,l1,l2,l3,l4,l5,l6,P1,R_new,G_new,B_new,densidadRuido)
	
	try:
		image_encrypt.save("encrypt.png")
	except:
		return "error parsing"	

	results = [image_encrypt, chaotic_sequences]
	return results

	
def substitution(Salida,M,N,t1,t2,t3,t4,t5,t6):
	
	Salida1 = generate_Li_sequences (Salida,M,N,t1,t2,t3,t4,t5,t6)
	Salida2 = generate_ELi_sequences (Salida1,M,N)
	Salida3 = sorted_Sequences_and_index(Salida2,M,N)
	Salida4 = quantify_index_sequences (Salida3,M,N)
	Salida5 = generate_diffusion_matrix(Salida4,M,N)
	
	return Salida5
	

def permutation(Salida,M,N,l1,l2,l3,l4,l5,l6,P1,R,G,B,densidadRuido):
	
	Salida1 = generate_LLi_sequences (Salida,M,N,l1,l2,l3,l4,l5,l6)
	Salida2 = generate_SLi_sequences (Salida1,M,N)
	Salida3 = generate_LS1_LS2_sequences(Salida2)
	Salida4 = div_LS1_LS3_sequences(Salida3,M,N)
	Salida5 = sorted_LSI_Sequences_and_index(Salida4,M,N)
	Salida6 = generate_scrambling_matrix (Salida5,M,N)
	Salida7 = shuffle_positions (P1,Salida6[0],M,N)
	Salida8 = obtain_components_I(Salida7)
	Salida9 = combinate_matrix_col_RGB (Salida8[0],Salida8[1],Salida8[2])
	Salida10 = scramble_positions(Salida9,Salida6[1],M,N)
	Salida11 = obtain_components_CI(Salida10)
	R_new = transform_value_to_image(R,Salida11[0],M,N)
	R_Noise = efectoSaltPepper(R_new, densidadRuido)
	G_new = transform_value_to_image(G,Salida11[1],M,N)
	G_Noise = efectoSaltPepper(G_new, densidadRuido)
	B_new = transform_value_to_image(B,Salida11[2],M,N)
	B_Noise = efectoSaltPepper(B_new, densidadRuido)
	image_encrypt = Image.merge('RGB',(R_Noise,G_Noise,B_Noise))
	return image_encrypt
	
'''Decription'''
			
def paso8(imagen,SM,M,N):
	
	
	a = range(0,3*M*N)
	
	for i in range(3*M):
		for j in range(N):
			indice = SM[i][j]*N + j
			a[indice] = imagen[i][j]
			
	
	P = np.array(a,int)
	P = P.reshape((3*M,N))
	
	return P
	
def paso5(imagen,SM,M,N):

	a = range(0,3*M*N)
	
	for i in range(M):
		for j in range(3*N):
			
			indice = SM[i][j] + i*3*N
			a[indice] = imagen[i][j]
			
	P = np.array(a,int)
	P = P.reshape((M,3*N))
	
	return P
	
def modify_pixel_value_2(R,G,B,EMi,M,N):

	
	for j in range(N-1,-1,-1):
		for i in range(M-1,-1,-1):
			if j == 0 :
				R_old = R.getpixel((i,0))
				R_aux = (R_old ^ EMi[0][0][i]) ^ EMi[3][0][i]
				R.putpixel((i,0),R_aux) 
				G_old = G.getpixel((i,0))
				G_aux = (G_old ^ EMi[1][0][i]) ^ EMi[4][0][i]
				G.putpixel((i,0),G_aux) 
				B_old = B.getpixel((i,0))
				B_aux = (B_old ^ EMi[2][0][i]) ^ EMi[5][0][i]
				B.putpixel((i,0),B_aux) 
			else:
				R_old = R.getpixel((i,j))
				R_aux = ((R_old ^ EMi[0][j][i]) ^ EMi[3][j][i]) ^ R.getpixel((i,j-1))
				R.putpixel((i,j),R_aux) 
				G_old = G.getpixel((i,j))
				G_aux = ((G_old ^ EMi[1][j][i]) ^ EMi[4][j][i]) ^ G.getpixel((i,j-1))
				G.putpixel((i,j),G_aux) 
				B_old = B.getpixel((i,j))
				B_aux = ((B_old ^ EMi[2][j][i]) ^ EMi[5][j][i]) ^ B.getpixel((i,j-1))
				B.putpixel((i,j),B_aux) 
				
	return R,G,B
	


	
def decrypt (a,b,c,k1,k2,InX,ruta):

	im = Image.open(ruta)
	im.load()
	M,N = im.size
	R,G,B = im.split()
	l = 1
	InXSequence = generate_l_sequences(a,b,c,k1,k2,InX,l,M,N)
	Salida2 = discard_old_l_values_of_sequences (InXSequence,l)
	R_aux = transform_image_to_value(R,M,N)
	G_aux = transform_image_to_value(G,M,N)
	B_aux = transform_image_to_value(B,M,N)
	
	P1 = combinate_matrix_col_RGB (R_aux,G_aux,B_aux)
	Rp, Gp, Bp = permutacionDec(Salida2,M,N,R_aux,G_aux,B_aux,R,G,B)
	
	image_decrypt = sustitucionDec(Rp,Gp,Bp,M,N,InXSequence)
	
	try:
		image_decrypt.save("decrypt.png")
	except:
		return "error parsing"	

	return image_decrypt
	
	
def permutacionDec(Imagen,M,N,Raux,Gaux,Baux,R,G,B):
	
	l1,l2,l3,l4,l5,l6 = [7,8,9,10,11,12]
	Salida1 = generate_LLi_sequences (Imagen,M,N,l1,l2,l3,l4,l5,l6)
	Salida2 = generate_SLi_sequences (Salida1,M,N)
	Salida3 = generate_LS1_LS2_sequences(Salida2)
	Salida4 = div_LS1_LS3_sequences(Salida3,M,N)
	Salida5 = sorted_LSI_Sequences_and_index(Salida4,M,N)
	SM = generate_scrambling_matrix (Salida5,M,N)
	Paso9 = combinate_matrix_col_RGB (Raux,Gaux,Baux)
	Paso8 = paso8(Paso9,SM[1],M,N)	
	Paso7 = obtain_components_CI(Paso8)
	Paso6 = combinate_matrix_RGB (Paso7[0],Paso7[1],Paso7[2],M,N)
	Paso5 = paso5(Paso6,SM[0],M,N)
	Paso4 = obtain_components_I(Paso5)
	R_new = transform_value_to_image(R,Paso4[0],M,N)
	G_new = transform_value_to_image(G,Paso4[1],M,N)
	B_new = transform_value_to_image(B,Paso4[2],M,N)

	return R_new,G_new,B_new
	
	
def sustitucionDec(R,G,B,M,N,chaotic_sequences):

	l = 1
	t1,t2,t3,t4,t5,t6 = [1,2,3,4,5,6]
	Salida2 = discard_old_l_values_of_sequences (chaotic_sequences,l)
	Salida3 = substitution(Salida2,M,N,t1,t2,t3,t4,t5,t6)
	Salida4 = Salida3[3],Salida3[4],Salida3[5],Salida3[0],Salida3[1],Salida3[2]
	R_new,G_new,B_new = modify_pixel_value_2(R,G,B,Salida4,M,N)
	img = Image.merge('RGB',(R_new,G_new,B_new))
	return img
