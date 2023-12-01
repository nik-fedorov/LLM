# LLM

Отчет о проделанной работе можно посмотреть в [wandb report](https://wandb.ai/nik-fedorov/LLM/reports/LLM-homework--Vmlldzo2MTMxNTMx).


## Синтез предложений (историй)

Подробно описано в ноутбуке `llm_hw.ipynb`


## Подготовка датасета и sentencepiece модели

```shell
bash download_tiny_stories.sh
python prepare.py
```


## Запуск тренировки

```shell
python main.py -c config.json
```
