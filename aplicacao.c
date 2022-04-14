#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
// Aloca memoria para uma
// matriz de dimencoes (N x M)
unsigned char** aloca(int m, int n){
    int i;
    unsigned char **M;
    M = (unsigned char**) malloc(n * sizeof(unsigned char*));
    for(i = 0; i < n; i++){
        M[i] = (unsigned char*) malloc(m * sizeof(unsigned char));
    }
    return M;
}
 
int convolution(int i, int j, short int mask[][3], unsigned char **image){
    int p, q, aux;
    int pixel_resultante = 0;
    i--; j--;
    aux = j;
    for(p = 0; p < 3; p++){
        for(q = 0; q < 3; q++){
            pixel_resultante += image[i][j] * mask[p][q];
            j++;
        }
        j = aux;
        i++;
    }
     //por estarmos utilizando uma matriz 3x3 de gauss, após a soma das multiplicações devemos dividir por 16
    return pixel_resultante/16;
}
 
int main()
{
    FILE *imagem; 
    FILE *nova_imagem;
    char nome_imagem[20];
    char nome_imagem_saida[20];
    char key[128];
    int i, j, m, n, max, pixel_resultante, threshold;
 
    printf("Digite o nome do arquivo PGM de entrada: ");
    scanf("%s", nome_imagem); // Le o nome do arquivo de entrada
    printf("Nome para arquivo de saida: ");
    scanf("%s", nome_imagem_saida);
  
    imagem = fopen(nome_imagem , "r") ; // Abre o arquivo no modo leitura
    if(imagem == NULL){ // Verificase o arquivo existe e foi aberto
        printf("Erro na abertura do arquivo %s\n", nome_imagem);
        return 0 ;
    }
 
    // Le cabecalho
    fscanf(imagem, "%s", key);
 
    // Imagens PGM tem "P5" na primeira linha
    if(strcmp(key,"P5") != 0){
        printf("Arquivo precisa ser PGM!\n") ;
        fclose(imagem);
        return 0 ;
    }
 
    //Próximos valores do cabeçalho após P5 são: numero de colunas, numero de linhas, e valor máximo.
    fscanf(imagem, "%d %d %d", &m, &n, &max) ;

 
    // Matriz para guardar a imagem original
    // Utilizando o método de bordas "pretas"
    unsigned char **original;
    original = aloca(m+2, n+2);

   
    // Leitura da imagem original
    for(i = 1; i < n+1; i++){
        for(j = 1; j < m+1; j++){
            fscanf(imagem, "%c", &original[i][j]);
        }
    }
 
    // matriz de convolução gaussiana
    short int mask[3][3] = {{1,  2,   1},
                            {2,  4,  2},
                            {1,  2,  1}};
 
 
    // abrir nova imagem em modo de escrita e "copiar" o cabeçalho da imagem original
    nova_imagem = fopen(nome_imagem_saida , "w");
    fprintf(nova_imagem,"P5\n%d %d\n %d\n", m, n, max); 
 
    for(i = 1; i <= n; i++){
        for(j = 1; j <= m; j++){ 
            pixel_resultante = convolution(i, j, mask, original); 

            fprintf(nova_imagem, "%c", (char) pixel_resultante);
        }
    }
 
    fclose(imagem);
    fclose(nova_imagem);
 
    for(i = 0; i < n+2; i++)
    {
        free(original[i]);
    }
    free(original);
    return 0;
}