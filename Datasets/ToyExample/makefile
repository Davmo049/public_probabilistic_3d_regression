build:
	mkdir build

build/crenderer.o: build c_src/crenderer.c
	gcc -c -fPIC -o build/crenderer.o -O3 c_src/crenderer.c

build/library.so: build/crenderer.o
	gcc -o build/library.so build/crenderer.o -shared
