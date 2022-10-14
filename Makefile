all:
	gcc -o done.out networks/mlp.c src/activations.h src/activations.c src/libs.h src/libs.c -lm
	./done.out


