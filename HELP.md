В проекте для хранения датасетов и моделей используется ClearML. Ниже представлена краткая инструкция по его использованию:     
* clearml-init - инициализация утилиты     
* clearml-data create --project eda --name all_datasets - создание списка отслеживаемых файлов        
* clearml-data add --files ./data - добавление файлов в список отслеживаемых      
* clearml-data close - отправка текущей версии списка на clearml сервер       
* clearml-data get --id 98dbde28e5114c8fa7bdc695eabe14e2 - скачать нужный коммит
* clearml-data search --name all_datasets - посмотреть дерево коммитов 
