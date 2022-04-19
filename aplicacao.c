#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void convolution(int n, int m, short int mask[][3], unsigned char *original, unsigned char *resultado){  
    int aux_i, aux_j;
    int aux;
    int pixel_resultante;
    int p, q; // for interno, da mascara 3x3
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            pixel_resultante = 0;
            aux_i = i; 
            aux_j = j;
            aux = j;
            for(p = 0; p < 3; p++){
                for(q = 0; q < 3; q++){
                    pixel_resultante += original[aux_i*n + aux_j] * mask[p][q];
                    aux_j++;
                }
                j = aux;
                aux_i++;
            }
            //por estarmos utilizando uma matriz 3x3 de gauss, após a soma das multiplicações devemos dividir por 16
            resultado[i*n + j] = pixel_resultante/16;
        }
    }
}

int main(int argc, char **argv){
    FILE *imagem;
    FILE *nova_imagem;
    char *nome_imagem;
    char *nome_imagem_saida;
    char key[128];
    int i, j, m, n, max, pixel_resultante, threshold;


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
    fscanf(imagem, "%d %d %d", &m, &n, &max) ;

    // Matriz para guardar a imagem original
    // Utilizando o método de bordas "pretas"
    unsigned char *original;
    int n_col = m+2;
    int n_row = n+2;
    original = (unsigned char*) malloc(n_row * n_col * sizeof(unsigned char*));


    // Matriz pra servir de buffer pra imagem resultado
    unsigned char *resultado;
    resultado = (unsigned char*) malloc(n * m *sizeof(unsigned char*));


    // Leitura da imagem original
    for(i = 1; i < n+1; i++){
        for(j = 1; j < m+1; j++){
            fscanf(imagem, "%c", &original[i*n + j]);
        }
    }

    // matriz de convolução gaussiana
    short int mask[3][3] = {{1,  2,   1},
                            {2,  4,  2},
                            {1,  2,  1}};


    // abrir nova imagem em modo de escrita e "copiar" o cabeçalho da imagem original
    nova_imagem = fopen(nome_imagem_saida , "w");
    fprintf(nova_imagem,"P5\n%d %d\n %d\n", m, n, max);


    convolution(n, m, mask, original, resultado);


    // escrever no arquivo resultado
    for(i = 0; i < n; i++){
        for(j = 0; j < m; j++){
            fprintf(nova_imagem, "%c", (char) resultado[i*n + j]);
        }
    }


    fclose(imagem);
    fclose(nova_imagem);

    free(original);
    free(resultado);

    return 0;
}