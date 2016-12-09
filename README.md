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

### Prerequisites

Todas as dependências necessárias estão presentes no arquivo requirements.txt e podem ser automaticamente baixadas e instaladas utilizando o seguinte comando:

```
$ conda create --name <env> --file requirements.txt
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
