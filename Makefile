all:
	gcc networks/mlp.c src/activations.h src/activations.c src/libs.h src/libs.c src/lossFunctions.h src/lossFunctions.c -lm
	./a.out


