# Reconhecimento da Língua Brasileira de Sinais usando sensores Eletromiográficos.

Este projeto utiliza os sensores eletromiográficos presentes no Myo Armband para coletar dados de sinais em Libras. Segue-se então a
segmentação e coleta de características a partir dos dados segmentados. Uma matriz de características é criada e utilizada para o 
treinamento e classificação utilizando Random Forest e Stratified K-Fold Cross-Validation.

## Getting Started

O fluxo básico que retorna a percentagem de acerto para um determinado grupo de letras está presente no arquivo main.py.
O comando abaixo deverá retornar a porcentagem de acerto para 15 letras se nada for alterado:

```
$ python main.py
```

O arquivo receiveData.py é responsável pela coleta dos dados. Ao se utilizar o comando abaixo, os dados do Myo irão ser coletados e salvos em um arquivo chamado paulo-A-1-emg.csv na pasta "data/paulo". Onde 'a' é a letra a ser realizada e 1 representa que essa é a primeira vez que essa letra é realizada.

```
$ python receiveData.py 'paulo' a 1
```

Para segmentar os dados de letra utilizamos os métodos presentes no arquivo segmentData.py. O método presente no arquivo main.py ~segmentFiles~
detecta os arquivos salvos dentro de ~data~, segmenta e em seguida salva-os dentro da pasta ~data_segmented~.

Para extrair as características dos dados segmentados, utilizamos o método getFeaturesEmg(data) do arquivo calculateFeatures.py. data representa os dados segmentados resultantes do passo anterior.

Por último, classificamos os dados utilizando o método classify do arquivo classification.py. Esse método recebe a matriz de características e retorna o índice de classificação para aquele grupo de letras.

### Prerequisites

Todas as dependências necessárias estão presentes no arquivo requirements.txt e podem ser automaticamente baixadas e instaladas utilizando o seguinte comando:

```
$ conda create --name <env> --file requirements.txt
```

## Built With

* [scikit-learn](scikit-learn.org/) 
* [pandas](pandas.pydata.org)
* [myo-python](https://github.com/NiklasRosenstein/myo-python)

## Authors

* **Paulo Fernandes** - *Initial work* - [Pernambucano](https://github.com/pernambucano)

## Agradecimentos

* Meus orientadores, Prof. Ricardo Prudêncio e Profa. Veronica Teichrieb
* Todos os que participaram direta ou indiretamente na produção deste projeto até agora.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
