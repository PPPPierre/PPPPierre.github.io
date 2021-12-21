---
title: 专业课 - 基于并行计算的代码优化
author: Stone SHI
date: 2021-10-11 15:14:00 +0200
categories: [Blogging, Study]
tags: [parallel_computing]
---

# 1. Compilation and executation

compile.

```sh
$ make
cc    -c -o io.o io.c
cc    -c -o transfo.o transfo.c
cc    -c -o cycles.o cycles.c
cc -o transform_image io.o transfo.o cycles.o
```

clean compilation output files.

```sh
$ make clean
rm -f *.o transform_image
rm -fr cmake-build-debug
```

The first executation.

enter `imagerie` folder, set the environmental variable

```sh
$ export IMAGES=../data
```

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
1174160151.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
1095196812.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
3534617694.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
909382992.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
622586028.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
851808507.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
875622380.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
212227580.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
410359449.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
486787547.000000 clock cycles.
TOTAL: 10172749140.000000 clock cycles.

real    0m20.266s
user    0m10.037s
sys     0m0.564s
```

After the verification, enter the `data` folder, clean the output files.

```sh
$ ./clean.sh
```

# 2. Optimisations 优化代码

À chaque modification, relancer le code et mesurer le gain obtenu. Garder trace de chaque version de votre code avec le gain obtenu. Recommendations:

1. Utiliser moins de fonctions

2. Mieux utiliser la mémoire (localité)

3. Utiliser moins de boucles

Vous pouvez vous aider de [perf](https://www.brendangregg.com/perf.html) (si vous êtes sous Linux) pour analyser plus finement le comportement de votre code

## Analysis of the source code 源代码解析

首先查看算法源代码，结合`data\transfo.txt`文件内容可以得出，这是一个将一张图片复制、进行像素值变换并提高亮度的算法。

```c
void copy (int w, int h, unsigned char *src, unsigned char *dest)
{
	int i,j;

  	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			dest[j * w + i] = src[j * w + i];
		}
	}
}

void light(int w, int h, unsigned char *img, unsigned char val)
{
	int i,j;
	unsigned char current;

	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			current = img[j * w + i];
			img[j * w + i] = (((int) current + val) > 255) ? 255 : current + val;
		}
	}
}

void curve(int w, int h, unsigned char *img, unsigned char *lut)
{
	int i,j;
  	unsigned char current;

  	for (i = 0; i < w; i++) {
  		for (j = 0; j < h; j++) {
  			current = img[j * w + i];
			img[j * w + i] = lut[current];
  		}
  	}
}

void transfo(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val)
{
  	copy(w, h, src, dest);
  	curve(w, h, dest, lut);
  	light(w, h, dest, val);
}
```

## 优化1

将三个函数压缩成一个函数，减少重复遍历

Combine three functions to one, reduece the total times of iteration.

```C
/*
void copy (int w, int h, unsigned char *src, unsigned char *dest)
{
	int i,j;

  	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			dest[j * w + i] = src[j * w + i];
		}
	}
}

void light(int w, int h, unsigned char *img, unsigned char val)
{
	int i,j;
	unsigned char current;

	for (i = 0; i < w; i++) {
		for (j = 0; j < h; j++) {
			current = img[j * w + i];
			img[j * w + i] = (((int) current + val) > 255) ? 255 : current + val;
		}
	}
}

void curve(int w, int h, unsigned char *img, unsigned char *lut)
{
	int i,j;
  	unsigned char current;

  	for (i = 0; i < w; i++) {
  		for (j = 0; j < h; j++) {
  			current = img[j * w + i];
			img[j * w + i] = lut[current];
  		}
  	}
}
*/

void loop_process(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val){
	int i,j;
  	unsigned char current;

  	for (i = 0; i < w; i++) {

		for (j = 0; j < h; j++) {
			// copy
			dest[j * w + i] = src[j * w + i];

			// curve
			current = dest[j * w + i];
			dest[j * w + i] = lut[current];

			// light
            current = dest[j * w + i];
			dest[j * w + i] = (((int) current + val) > 255) ? 255 : current + val;
		}
  	}
}

void transfo(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val)
{
	/*
  	copy(w, h, src, dest);
  	curve(w, h, dest, lut);
  	light(w, h, dest, val);
	*/

	loop_process(w, h, src, dest, lut, val);

}
```

Test

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
718480772.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
668396133.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
2273278456.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
590257783.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
336672412.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
440642125.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
520065354.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
127555428.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
235020610.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
292524428.000000 clock cycles.
TOTAL: 6202893501.000000 clock cycles.

real    0m17.814s
user    0m7.847s
sys     0m0.653s
```

2-3 seconds faster

## 优化2

We can compress two `for` into one `for` loop.

显然两个 `for` 循环可以压缩成一个 `for` 循环。

The improved codes are shown below.

改完代码如下：

```c
void loop_process(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val){
	int n = w * h;
  	unsigned char current;

	int i;

  	for (i = 0; i < n; i++) {
		// copy
		dest[i] = src[i];

		// curve
		current = dest[i];
		dest[i] = lut[current];

		// light
        current = dest[i];
		dest[i] = (((int) current + val) > 255) ? 255 : current + val;
  	}
}
```

Test

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
241626861.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
207095052.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
688837133.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
188729000.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
110668988.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
153436717.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
216416799.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
48847379.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
87362807.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
110621170.000000 clock cycles.
TOTAL: 2053641906.000000 clock cycles.

real    0m17.044s
user    0m6.391s
sys     0m0.845s
```

1 sec faster.

## 优化3

Improve the algorithm:

算法显然可以再优化，优化完结果：

```c
void loop_process(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val){
	int n = w * h;
  	unsigned char tmp;

	int i;

  	for (i = 0; i < n; i++) {
		current = src[i];
		dest[i] = (((int) current + val) > 255) ? 255 : current + val;
  	}
}
```

Test

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
107953226.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
88947604.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
336525503.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
78052902.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
50549115.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
63703490.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
75967615.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
22521228.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
39134269.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
43233903.000000 clock cycles.
TOTAL: 906588855.000000 clock cycles.

real    0m16.410s
user    0m5.832s
sys     0m0.806s
```

1s faster.

## 优化4 Loop unrolling 循环展开

使用循环展开的技术来压缩时间。

优化后代码：

```c transfo.c
void loop_process(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val){
	int n = w * h;
  	unsigned char current_0, current_1, current_2, current_3;

	int i;

  	for (i = 0; i < n-3; i+=4) {
		current_0 = lut[src[i]];
		current_1 = lut[src[i+1]];
		current_2 = lut[src[i+2]];
		current_3 = lut[src[i+3]];
		dest[i] = (((int) current_0 + val) > 255) ? 255 : current_0 + val;
		dest[i+1] = (((int) current_1 + val) > 255) ? 255 : current_1 + val;
		dest[i+2] = (((int) current_2 + val) > 255) ? 255 : current_2 + val;
		dest[i+3] = (((int) current_3 + val) > 255) ? 255 : current_3 + val;
  	}
}
```

Test

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
93511300.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
74312291.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
295375831.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
65381472.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
41186202.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
53358892.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
63093263.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
18141256.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
33467929.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
36029574.000000 clock cycles.
TOTAL: 773858010.000000 clock cycles.

real    0m14.911s
user    0m4.991s
sys     0m0.510s
```

性能提高。

Better performance.

# 优化5

`lut` saturation.

可以对`lut`变换的数组先加`val`进行饱和判断操作，然后遍历进行变换，这样会减少更多循环的次数以及`if`判断的次数。

```c
void loop_process(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val){
	int n = w * h;

	int i;

	for (i = 0; i <= 255; i ++) {
		lut[i] = (((int) lut[i] + val) > 255) ? 255 : lut[i] + val;
  	}

  	for (i = 0; i < n-3; i+=4) {
		dest[i] = lut[src[i]];
		dest[i+1] = lut[src[i+1]];
		dest[i+2] = lut[src[i+2]];
		dest[i+3] = lut[src[i+3]];
  	}
}
```

Test

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
62751136.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
50819186.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
220661554.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
43584822.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
27588220.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
37273392.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
42147550.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
20684776.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
22268526.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
24155182.000000 clock cycles.
TOTAL: 551934344.000000 clock cycles.

real    0m15.721s
user    0m5.409s
sys     0m0.593s
```

反而下降了，说明这个方法没有改变太多。也可能是电脑运行久了变热了运算速度下降了。

The problem is that the system I used is wsl2 and the file path is in `/mnt/c/` which will make the reading process cost more time.

# Parallélisation(s) 并行计算

Voici des pistes de parallélisation, qui peuvent être complémentaires:

1. **Vectorisation**: utiliser les instructions vectorielles, soit en optimisant l'assembleur "à la main", soit en utilisant les intrinsics x86 présents dans [GCC](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html), soit en utilisant les [bons paramètres du compilateur](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html). Cette étape nécessitera certainement d'aider le compilateur en lui présentant des boucles "facilement" vectorisables. Vérifiez également que GCC génère du code [correspondant à votre machine](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html). La documentation [Intel](https://software.intel.com/content/www/us/en/develop/download/intel-64-and-ia-32-architectures-optimization-reference-manual.html) sur l'optimisation de code pour architecture x86-64 donne de nombreuses pistes d'optimisation.
2. **Multithreads**: utiliser [OpenMP](https://connect.ed-diamond.com/GNU-Linux-Magazine/glmf-122/decouverte-de-la-programmation-parallele-avec-openmp) pour obtenir une version du code utilisant plusieurs threads. Mesurer le gain obtenu en fonction du nombre de threads utilisés.
3. **Multiprocessus**: modifier le code en utilisant plusieurs processus. On peut imaginer ici que l'on utilise des processus en parallèle où chaque processus s'occuperait d'une image. Ainsi, on pourra cherche ici à améliorer le temps pris par le traitement des dix images à la suite. Se rappeler de fork().
4. **GPU**: utiliser CUDA afin d'obtenir une version du code fonctionnant sur un processeur graphique.

## GCC loop distribution

在 `CMakeLists.txt`里面加上一句

`SET ( CMAKE_CXX_FLAGS "-fopenmp" )`

```CMakeLists.txt
project(transform_image)

SET ( CMAKE_CXX_FLAGS "-fopenmp" )

add_executable(transform_image cycles.c cycles.h io.c transfo.c transfo.h)
```

然后在要加速的`for`循环前加一行`#pragma omp parallel for`

```c
void loop_process(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val){
	int n = w * h;

	int i;

	#pragma omp parallel for
	for (i = 0; i <= 255; i ++) {
		lut[i] = (((int) lut[i] + val) > 255) ? 255 : lut[i] + val;
  	}

	#pragma omp parallel for
  	for (i = 0; i < n-3; i+=4) {
		dest[i] = lut[src[i]];
		dest[i+1] = lut[src[i+1]];
		dest[i+2] = lut[src[i+2]];
		dest[i+3] = lut[src[i+3]];
  	}
}
```

测试

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
71895122.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
57267568.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
237394217.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
43233580.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
27222220.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
52180870.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
42858312.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
12074296.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
21542948.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
24088950.000000 clock cycles.
TOTAL: 589758083.000000 clock cycles.

real    0m14.813s
user    0m5.547s
sys     0m0.430s
```

有所效果，但不明显。

在不使用循环展开的情况下检查效果：

```c
void loop_process(int w, int h, unsigned char *src, unsigned char *dest, unsigned char *lut, unsigned char val){
	int n = w * h;

	int i;

	#pragma omp parallel for
	for (i = 0; i <= 255; i ++) {
		lut[i] = (((int) lut[i] + val) > 255) ? 255 : lut[i] + val;
  	}

	#pragma omp parallel for
  	for (i = 0; i < n; i++) {
		dest[i] = lut[src[i]];
  	}
}
```

```sh
$ time ./transform_image $IMAGES/transfo.txt
image1.pgm courbe1.amp 5 image1_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
90885426.000000 clock cycles.
image2.pgm courbe2.amp 5 image2_t.pgm
image2.pgm: 5227 x 3515 = 18372905 pixels
73543448.000000 clock cycles.
image3.pgm courbe3.amp 5 image3_t.pgm
image3.pgm: 6660 x 9185 = 61172100 pixels
283044717.000000 clock cycles.
image4.pgm courbe4.amp 4 image4_t.pgm
image4.pgm: 3381 x 4914 = 16614234 pixels
63834405.000000 clock cycles.
image5.pgm courbe5.amp 7 image5_t.pgm
image5.pgm: 3226 x 3255 = 10500630 pixels
43076592.000000 clock cycles.
image6.pgm courbe6.amp 6 image6_t.pgm
image6.pgm: 3677 x 3677 = 13520329 pixels
52240602.000000 clock cycles.
image7.pgm courbe7.amp 9 image7_t.pgm
image7.pgm: 3264 x 4896 = 15980544 pixels
62215935.000000 clock cycles.
image8.pgm courbe8.amp 5 image8_t.pgm
image8.pgm: 1757 x 2636 = 4631452 pixels
17739868.000000 clock cycles.
image9.pgm courbe9.amp 7 image9_t.pgm
image9.pgm: 2498 x 3330 = 8318340 pixels
32985879.000000 clock cycles.
image10.pgm courbe10.amp 9 image10_t.pgm
image10.pgm: 3024 x 3024 = 9144576 pixels
35069814.000000 clock cycles.
TOTAL: 754636686.000000 clock cycles.

real    0m14.513s
user    0m5.108s
sys     0m0.538s
```

可以看见和前者相差不大，循环展开提升的效果在并行计算的情况下被弱化了。

## Multithreads - OpenMP

OpenMP 官方上介绍的方法和上一个差不多，应该也是基于 gcc 编译过程中的 loop disstribution 的技术。

## Multiprocessus - fork()

fork() functions.

使用消息队列完成进程的内部通信

导入两个进程相关的头文件 `<unistd.h>` `<sys/wait.h>`

以及一个用于进程间信息交换的头文件 `<sys/ipc.h>`

```c
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
```

```c io.c
void run_transfo_file(FILE *tf)
{
	char source[FNMAX];
	char curve[FNMAX];
	char dest[FNMAX];
	int light;
	
	pid_t child_pid, wpid;
	int status = 0;
	double total;

	while (fscanf(tf, "%s %s %d %s", source, curve, &light, dest) == 4) {
		child_pid = fork();
		if (child_pid == 0) {
			int res;
			printf("pid: %d %s %s %d %s\n",getpid(), source, curve, light, dest);
			total += transform_image(source, curve, light, dest);
			exit(0);
		}
	}
	
	while ((wpid = wait(&status)) > 0);

	printf("TOTAL: %f clock cycles.\n", total);
}
```

Test

```sh
$ time ./transform_image $IMAGES/transfo.txt
pid: 5076 image1.pgm courbe1.amp 5 image1_t.pgm
pid: 5077 image2.pgm courbe2.amp 5 image2_t.pgm
pid: 5078 image3.pgm courbe3.amp 5 image3_t.pgm
pid: 5079 image4.pgm courbe4.amp 4 image4_t.pgm
pid: 5080 image5.pgm courbe5.amp 7 image5_t.pgm
pid: 5081 image6.pgm courbe6.amp 6 image6_t.pgm
pid: 5082 image7.pgm courbe7.amp 9 image7_t.pgm
pid: 5083 image8.pgm courbe8.amp 5 image8_t.pgm
pid: 5084 image9.pgm courbe9.amp 7 image9_t.pgm
pid: 5085 image10.pgm courbe10.amp 9 image10_t.pgm
image1.pgm: 5617 x 3684 = 20693028 pixels
image2.pgm: 5227 x 3515 = 18372905 pixels
image3.pgm: 6660 x 9185 = 61172100 pixels
image4.pgm: 3381 x 4914 = 16614234 pixels
image5.pgm: 3226 x 3255 = 10500630 pixels
image7.pgm: 3264 x 4896 = 15980544 pixels
image6.pgm: 3677 x 3677 = 13520329 pixels
image8.pgm: 1757 x 2636 = 4631452 pixels
image10.pgm: 3024 x 3024 = 9144576 pixels
image9.pgm: 2498 x 3330 = 8318340 pixels
21427730.000000 clock cycles.
37286814.000000 clock cycles.
38725273.000000 clock cycles.
45551827.000000 clock cycles.
58591918.000000 clock cycles.
69907987.000000 clock cycles.
88265978.000000 clock cycles.
68358251.000000 clock cycles.
73210363.000000 clock cycles.
187752982.000000 clock cycles.
TOTAL: 0.000000 clock cycles.

real    0m7.214s
user    0m7.764s
sys     0m1.871s
```

The value TOTAL in the result is 0, actually we need to realize a communication between parend process and son processes.