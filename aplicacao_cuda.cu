#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__global__
void convolution(int altura, int largura, short int *mask, unsigned char *original, unsigned char *resultado){  
    int i = blockIdx.y * blockDim.y + threadIdx.x;
    int j = blockIdx.x * blockDim.x + threadIdx.y;
    int aux_i, aux_j;
    int pixel_resultante;
    int p, q; // for interno, da mascara 3x3

    if (i >= altura || j >= largura){
        return;
    }

    pixel_resultante = 0;
    aux_i = i; 
    for(p = 0; p < 5; p++){
        aux_j = j;
        for(q = 0; q < 5; q++){
            pixel_resultante += original[aux_i*largura + aux_j] * mask[p*5 + q];
            aux_j++;
        }
        aux_i++;
    }
    //por estarmos utilizando uma matriz 5x5 de gauss, após a soma das multiplicações devemos dividir por 273
    resultado[i*largura + j] = pixel_resultante/273;
}

int main(int argc, char **argv){
    FILE *imagem;
    FILE *nova_imagem;
    char *nome_imagem;
    char *nome_imagem_saida;
    char key[128];
    int i, j, largura, altura, max;
    float etime;
    struct timespec inic, fim;

    if (argc != 3){
        printf("Erro, o programa deve receber o nome da imagem de entrada \n"
                "e um nome para imagem de saida.");
        return 0;
    }

    nome_imagem = argv[1];
    nome_imagem_saida = argv[2];

    imagem = fopen(nome_imagem , "r") ; // Abre o arquivo no modo leitura
    if(imagem == NULL){ // Verificase o arquivo existe e foi aberto
        printf("Erro na abertura do arquivo %s\n", nome_imagem);
        return 0;
    }

    // Le cabecalho
    fscanf(imagem, "%s", key);

    // Imagens PGM tem "P5" na primeira linha
    if(strcmp(key,"P5") != 0){
        printf("Arquivo precisa ser PGM!\n") ;
        fclose(imagem);
        return 0;
    }

    //Próximos valores do cabeçalho após P5 são: numero de colunas, numero de linhas, e valor máximo.
    fscanf(imagem, "%d %d %d", &largura, &altura, &max) ;

    // Matriz para guardar a imagem original
    // Utilizando o método de bordas "pretas"
    unsigned char *original, *d_original;
    int n_col = largura+2;
    int n_row = altura+2;

    cudaMalloc(&d_original, n_row * n_col * sizeof(unsigned char*));
    original = (unsigned char*) malloc(n_row * n_col * sizeof(unsigned char*));


    // Matriz pra servir de buffer pra imagem resultado
    unsigned char *resultado, *d_resultado;
    resultado = (unsigned char*) malloc(altura * largura *sizeof(unsigned char*));
    cudaMalloc(&d_resultado, n_row * n_col * sizeof(unsigned char*));


    // Leitura da imagem original
    for(i = 1; i < altura+1; i++){
        for(j = 1; j < largura+1; j++){
            fscanf(imagem, "%c", &original[i*largura + j]);
        }
    }

    cudaMemcpy(d_original, original, altura * largura *sizeof(unsigned char*), cudaMemcpyHostToDevice);

    // matriz de convolução gaussiana
    // gaus 3x3
    // short int mask[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    
    //gaus 5x5
    short int mask[25] = {1,  4,  7,  4, 1, 
                          4, 16, 26, 16, 4, 
                          7, 26, 41, 26, 7, 
                          4, 16, 26, 16, 4, 
                          1,  4,  7,  4, 1};


    short int *d_mask;
    cudaMalloc(&d_mask, 5 * 5 * sizeof(short int*));
    cudaMemcpy(d_mask, mask, 5 * 5 * sizeof(short int*), cudaMemcpyHostToDevice);


    // para calcular uma imagem inteira, 256x256 precisamos de um grid 8x8 (8x32 =256)
    // o x indica o calculo dos pixeis na horizontal e o y na vertical
    // como o valor máximo de threads por bloco são travados em 1024 manteremos o dim3 block com valores travados (em 1024) e 
    // alteraremos o valor do grid conforme o MxN pego na leitura do arquivo.
    dim3 grid(largura/32,altura/32);
    dim3 block(32,32);
    

    clock_gettime(CLOCK_REALTIME, &inic);

    convolution<<<grid,block>>>(altura, largura, d_mask, d_original, d_resultado);
    cudaMemcpy(resultado, d_resultado, altura * largura * sizeof(unsigned char*), cudaMemcpyDeviceToHost);
    const char *string_error = cudaGetErrorString(cudaGetLastError());
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_REALTIME, &fim);

    // tempo decorrido: elapsed time
    etime = (fim.tv_sec + fim.tv_nsec/1000000000.) - 
            (inic.tv_sec + inic.tv_nsec/1000000000.) ;

    printf("Tempo da convolução: %lf\n", etime);
    printf("Altura: %d\tLargura: %d\n", altura ,largura);
    

    if(strcmp(string_error,"no error") != 0){
      printf("Device Variable Copying:\t%s\n", string_error);
    }
    // abrir nova imagem em modo de escrita e "copiar" o cabeçalho da imagem original
    nova_imagem = fopen(nome_imagem_saida , "w");
    fprintf(nova_imagem,"P5\n%d %d\n %d\n", largura, altura, max);

    // escrever no arquivo resultado
    for (i = 0; i < altura ; i++){
        for (j = 0; j < largura; j++){
            fprintf(nova_imagem, "%c", resultado[i*largura + j]);
        }
    }
    
    
    //fputs(resultado, nova_imagem);



    fclose(imagem);
    fclose(nova_imagem);

    free(original);
    free(resultado);

    cudaFree(d_mask);
    cudaFree(d_original);
    cudaFree(d_resultado);

    return 0;
}