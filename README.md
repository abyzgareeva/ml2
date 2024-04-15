# Рекомандательная система музыки

# Входные данные: 
Название музыкального трека

# Результат: 
Список рекомендуемых песен

# Датасет: 
Датасет состоит из столбцов instance_id,	artist_name, track_name,	popularity,	acousticness,	danceability,	duration_ms	energy,	instrumentalness,	liveness,	loudness,	speechiness,	tempo,	valence,	music_genre (идентификатор экземпляра, имя исполнителя, название трека, популярность, акустичность, танцевальность, длительность, энергетика, инструментальность, живость, громкость, выразительность, темп, валентность, музыкальный жанр)

# Алгоритм
Для получения списка рекомендаций использовался Метод k-ближайших соседей.

# Полученные результаты: 
Для оценки полученных результатов, я попросила воспользоваться системой рекомендаций 20 человек и спросила сколько треков из предложенных им понравились. 

На первой итерации я получила следующий резулультаты: В среднем 72% рекомендуемых треков были оценены как понравившееся. 

Для улучшения полученного результата я увиличила колличество соседних элеметов, используемых для запросов. В итоге в среднем 80,5% рекомендуемых исполнителей были отмечены как понравившиеся. 

# Приложение: 
![image](https://github.com/abyzgareeva/ml2/assets/61008851/677fb681-8c39-486b-9a80-0f866924f069)
![image](https://github.com/abyzgareeva/ml2/assets/61008851/8856aebf-2e40-4ac6-b2ec-6b6c8b959c30)
