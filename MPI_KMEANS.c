//============================================================================
// Name:			KMEANS.c
// Compilacion:	gcc KMEANS.c -o KMEANS -lm
//
// Autores:
// Fernando San Jose Dominguez
// Mario Pereda Puyo
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <stdbool.h>
//Constantes
#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
Muestra el correspondiente errro en la lectura de fichero de datos
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tEl fichero %s contiene demasiadas columnas.\n", filename);
			fprintf(stderr,"\tSe supero el tamano maximo de columna MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error leyendo el fichero %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error escibiendo en el fichero %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

/* 
Lectura del fichero para determinar el numero de filas y muestras (samples)
*/
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}	    
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else
	{
    	return -2;
	}
}

/* 
Carga los datos del fichero en la estructra data
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {         
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/* 
Escribe en el fichero de salida la clase a la que perteneces cada muestra (sample)
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}

/*
Copia el valor de los centroides de data a centroids usando centroidPos como
mapa de la posicion que ocupa cada centroide en data
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
Calculo de la distancia euclidea
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/*
Funcion de clasificacion, asigna una clase a cada elemento de data
*/
int classifyPoints(float *data, float *centroids, int *classMap, int lines, int samples, int K){
	int i,j;
	int class;
	float dist, minDist;
	int changes=0;
	for(i=0; i<lines; i++)
	{
		class=1;
		minDist=FLT_MAX;
		for(j=0; j<K; j++)
		{
			dist=euclideanDistance(&data[i*samples], &centroids[j*samples], samples);

			if(dist < minDist)
			{
				minDist=dist;
				class=j+1;
			}
		}
		if(classMap[i]!=class)
		{
			changes++;
		}
		classMap[i]=class;
	}
	return(changes);
}

/*
Recalcula los centroides a partir de una nueva clasificacion
*/
float recalculateCentroids(float *data, float *centroids, int *classMap, int lines, int samples, int K,int rank){
	int class, i, j;
	int *pointsPerClass;
	pointsPerClass=(int*)calloc(K,sizeof(int));
	float *auxCentroids;
	auxCentroids=(float*)calloc(K*samples, sizeof(float));
	float *distCentroids;
	distCentroids=(float*)malloc(K*sizeof(float));
	float maxDist=FLT_MIN;
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Error alojando memoria\n");
		MPI_Finalize();
		exit(-4);
	}

	//pointPerClass: numero de puntos clasificados en cada clase
	//auxCentroids: media de los puntos de cada clase 
	
	for(i=0; i<lines; i++) 
	{
		class=classMap[i];
		pointsPerClass[class-1] = pointsPerClass[class-1] +1;
		for(j=0; j<samples; j++){
			auxCentroids[(class-1)*samples+j] += data[i*samples+j];
		}
	}
	
	int *pointsPerClassTotal=(int*)calloc(K,sizeof(int));
	float *auxCentroidsTotal=(float*)calloc(K*samples, sizeof(float));

	MPI_Reduce(auxCentroids,auxCentroidsTotal,K*samples,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Reduce(pointsPerClass,pointsPerClassTotal,K,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	
	if(rank==0){
	
	for(i=0; i<K; i++) 
	{

		for(j=0; j<samples; j++){
			auxCentroidsTotal[i*samples+j] /= pointsPerClassTotal[i];
		}
	}
	


	
	for(i=0; i<K; i++){
		distCentroids[i]=euclideanDistance(&centroids[i*samples], &auxCentroidsTotal[i*samples], samples);
		if(distCentroids[i]>maxDist) {
			maxDist=distCentroids[i];
		}
	}
	memcpy(centroids, auxCentroidsTotal, (K*samples*sizeof(float)));
	}
	
	MPI_Bcast(centroids, K*samples, MPI_FLOAT, 0, MPI_COMM_WORLD);;
	
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);
	free(auxCentroidsTotal);
	free(pointsPerClassTotal);
	return(maxDist);
}



int main(int argc, char* argv[])
{

	MPI_Init(&argc, &argv);	
	int rank, nprocs;
	//START CLOCK***************************************
	double t_begin, t_end;
	t_begin = MPI_Wtime();
	//**************************************************
	/*
	 * PARAMETROS
	 *
	 * argv[1]: Fichero de datos de entrada
	 * argv[2]: Numero de clusters
	 * argv[3]: Numero maximo de iteraciones del metodo. Condicion de fin del algoritmo
	 * argv[4]: Porcentaje minimo de cambios de clase. Condicion de fin del algoritmo.
	 * 			Si entre una iteracion y la siguiente el porcentaje cambios de clase es menor que
	 * 			este procentaje, el algoritmo para.
	 * argv[5]: Precision en la distancia de centroides depuesde la actualizacion
	 * 			Es una condicion de fin de algoritmo. Si entre una iteracion del algoritmo y la 
	 * 			siguiente la distancia maxima entre centroides es menor que esta precsion el
	 * 			algoritmo para.
	 * argv[6]: Fichero de salida. Clase asignada a cada linea del fichero.
	 * */
	if(argc !=  7)
	{
		fprintf(stderr,"EXECUTION ERROR KMEANS Iterative: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}
	
	//Inicializaciones MPI
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int lines = 0, samples= 0; 
	float *data=NULL;
	int error = 0;	
	if(rank==0){
	//Lectura de los datos de entrada
	// lines = numero de puntos;  samples = numero de dimensiones por punto	
	error = readInput(argv[1], &lines, &samples);

	if(error != 0)
	{
		showFileError(error,argv[1]);
		MPI_Finalize();
		exit(error);
	}
	
	data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Error alojando memoria\n");
		MPI_Finalize();
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		MPI_Finalize();
		exit(error);
	}
	}
//Pasamos lines y samples a los procesos
	MPI_Bcast(&lines,1,MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&samples,1,MPI_INT, 0, MPI_COMM_WORLD);
	
//calculamos tamaños y restos de los procesos
	int mySize=lines/nprocs;
	int resto = lines%nprocs;
	if(rank<resto){
		mySize++;
	}

	MPI_Barrier(MPI_COMM_WORLD);
//creamos la matriz donde se alojaran los datos de cada proceso	
	float *myMatrix=(float *) calloc(mySize*samples,sizeof(float));//vector qeu recibe cada proceso
	int myCount = mySize*samples;//cada cantidad de datos de cada proceso

// prametros del algoritmo. La entrada no esta valdidada
	int K=atoi(argv[2]); 
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);


	int *centroidPos=NULL;
	float *centroids=NULL;
	int *classMap=NULL;
	
	//poscion de los centroides en data
	centroidPos = (int*)calloc(K,sizeof(int));
	centroids = (float*)calloc(K*samples,sizeof(float));
	classMap = (int*)calloc(mySize,sizeof(int));//cada proceso tiene su classMap

	float distCent;
	//Otras variables
    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Error alojando memoria\n");
		MPI_Finalize();
		exit(-4);
	}
	int it=0;
	int changes = 0;
	
	//inicializacion de saltos y contadores de datos
	  int *count=NULL;
	  int *jumps=NULL;
	
	// Centroides iniciales
	if(rank ==0){
	
	srand(0);
	int i=0;
	for(i=0; i<K; i++) 
		centroidPos[i]=rand()%lines;
	
	//Carga del array centroids con los datos del array data
	//los centroides son puntos almacenados en data
	initCentroids(data, centroids, centroidPos, samples, K);

	//rellenamos los saltos y numero de elementos que tendran los buffers de los procesos
        count=(int *)malloc(nprocs*sizeof(int));
    	jumps=(int *)malloc(nprocs*sizeof(int));
	
	int position=0;
	for(i=0;i<nprocs;i++){
            jumps[i]=position*samples;
            int size=lines/nprocs;
            int restos=lines%nprocs;
            size=(i<restos)?size+1:size;
            count[i]=size*samples;
            position=position+size;
	}


	// Resumen de datos caragos
	printf("\n\tFichero de datos: %s \n\tPuntos: %d\n\tDimensiones: %d\n", argv[1], lines, samples);
	printf("\tNumero de clusters: %d\n", K);
	printf("\tNumero maximo de iteraciones: %d\n", maxIterations);
	printf("\tNumero minimo de cambios: %d [%g%% de %d puntos]\n", minChanges, atof(argv[4]), lines);
	printf("\tPrecision maxima de los centroides: %f\n", maxThreshold);
	}
	//END CLOCK*****************************************
	t_end = MPI_Wtime();
	
	if(rank==0){
	printf("\nAlojado de memoria: %f segundos\n",(t_end - t_begin));
	fflush(stdout);
	}

	//**************************************************
	//START CLOCK***************************************
	t_begin = MPI_Wtime();
	//**************************************************
	
	//Pasamos la matriz de posiciones y centroides
	MPI_Bcast(centroidPos,K,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(centroids,K*samples,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	//Pasamos la matriz de datos a cada proceso
	//Cada proceso tendra su inicio de memoria con saltos
	MPI_Scatterv(data,count,jumps,MPI_FLOAT,myMatrix,myCount,MPI_FLOAT,0,MPI_COMM_WORLD);
	bool stopProcess=false;//Condicio nde parada de todos los procesos
	
	//Bucle principal	
	do{	
		it++;
		//Calcula la distancia desde cada punto al centroide
		//Asigna cada punto al centroide mas cercano
		int myChanges=classifyPoints(myMatrix, centroids, classMap, mySize, samples, K);	
		MPI_Reduce(&myChanges,&changes,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
		//Recalcula los centroides: calcula la media dentro de cada centoide
		float myDistCent=recalculateCentroids(myMatrix, centroids, classMap, mySize, samples,K,rank);
		MPI_Reduce(&myDistCent,&distCent,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
		
		if(rank==0){
			if((changes<=minChanges)||(it>=maxIterations)||(distCent<=maxThreshold)){
				stopProcess = true;
			}
			printf("\n[%d] Cambios de cluster: %d\tMax. dist. centroides: %f",it, changes, distCent);
		}

		MPI_Bcast(&stopProcess,1,MPI_C_BOOL,0,MPI_COMM_WORLD);

	} while(!stopProcess);

		
	int *classMapFinal = (int*)calloc(lines,sizeof(int));//classMap final donde verten todos los procesos su classMap propio al proceso 0
	
	//Declaramos los nuevos saltos y contadores de datos para enviar al proceso 0	
	
	
	//Calculamos los saltos y datos totales de cada proceso
	if(rank==0){
	int newPosition=0;
	for(int i=0;i<nprocs;i++){	
		jumps[i]=newPosition;
		int sizeSend=lines/nprocs;
		int restosSend=lines%nprocs;
		sizeSend=(i<restosSend)?sizeSend+1:sizeSend;
		count[i]=sizeSend;
		newPosition = newPosition + sizeSend;
	}	
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	//Volcamos todos classMaps al proceso 0
	MPI_Gatherv(classMap,mySize,MPI_INT,classMapFinal,count,jumps,MPI_INT,0,MPI_COMM_WORLD);
	
	//rango 0 imprime los resultados finales de finalizacion
	if(rank==0){
	//Condiciones de fin de la ejecucion
	if (changes<=minChanges) {
		printf("\n\nCondicion de parada: Numero minimo de cambios alcanzado: %d [%d]",changes, minChanges);
	}
	else if (it>=maxIterations) { 
		printf("\n\nCondicion de parada: Numero maximo de iteraciones alcanzado: %d [%d]",it, maxIterations);
	}
	else{
		printf("\n\nCondicion de parada: Precision en la actualizacion de centroides alcanzada: %g [%g]",distCent, maxThreshold);
	}	

	//Escritura en fichero de la clasificacion de cada punto
	error = writeResult(classMapFinal, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		MPI_Finalize();
		exit(error);
	}
	}
	//END CLOCK*****************************************
	t_end = MPI_Wtime();
	if(rank==0){
	printf("\nComputacion: %f segundos", (t_end - t_begin));
	fflush(stdout);
	}
	//**************************************************
	//START CLOCK***************************************
	t_begin = MPI_Wtime();
	//**************************************************


	//Liberacion de la memoria dinamica
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(jumps);
	free(count);
	free(myMatrix);
	free(classMapFinal);
	//END CLOCK*****************************************
	t_end = MPI_Wtime();

	if(rank==0){
	printf("\n\nLiberacion: %f segundos\n", (t_end - t_begin));
	fflush(stdout);
	}
	//***************************************************/
		
	MPI_Finalize();
	return 0;
}
