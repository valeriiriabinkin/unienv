# Задание 2 (Изучите алгоритм REINFORCE Chapter11/
02_cartpole_reinforce.py. Исследуйте влияние гиперпараметров на среднее количество шагов сходимости.)
Во втором задании мы знакомимся с реализацией алгоритма REINFORCE в среде cartpole.
У нас представлены следующие параметры:Гиперпараметры: Определяются гиперпараметры алгоритма, такие как коэффициент дисконтирования GAMMA, скорость обучения LEARNING_RATE и количество эпизодов для обучения EPISODES_TO_TRAIN.
Для начала выясним, что в теории влечет за собой изменение тех или иных гиперпараметров:
GAMMA (коэффициент дисконтирования):

Увеличение GAMMA приведет к увеличению влияния будущих вознаграждений на текущие действия. Это может увеличить стабильность обучения и способствовать более долгосрочному планированию агента.
Уменьшение GAMMA приведет к уменьшению влияния будущих вознаграждений на текущие действия. Это может привести к более короткосрочному планированию агента и более чувствительному к непосредственным результатам.

LEARNING_RATE (скорость обучения):

Увеличение LEARNING_RATE может ускорить сходимость, особенно в начале обучения. Однако слишком высокая скорость обучения может привести к осцилляциям или расхождению.
Уменьшение LEARNING_RATE может увеличить стабильность обучения и сделать процесс более надежным, но может потребоваться больше времени для сходимости.

EPISODES_TO_TRAIN (количество эпизодов для обучения):

Увеличение EPISODES_TO_TRAIN может привести к более стабильному и точному обучению, так как агенту предоставляется больше данных для обучения.
Однако это также может увеличить время обучения и требования к вычислительным ресурсам.

У нас будет 3 испытания, в каждом из которов 5 повторений: запуск на стандартных параметрах, запуск с lr = 0.005
и запуск с lr = 0.005 и episodes_to_train = 16.

В наш SummaryWriter записывается:
reward: Записывается награда (reward), полученная агентом в каждом шаге обучения. Это позволяет отслеживать изменение награды во времени и оценить процесс обучения.

reward_100: Записывается среднее значение награды за последние 100 эпизодов. Это помогает оценить стабильность обучения и его прогресс на протяжении времени.

episodes: Записывается количество завершенных эпизодов на каждом шаге обучения. Это помогает отслеживать, сколько эпизодов агент завершил к моменту текущего шага.

Реузльтаты со стандартными гиперпараметрами:
![Screenshot_1.jpg](imgs_sem_6%2FScreenshot_1.jpg)
![Screenshot_2.jpg](imgs_sem_6%2FScreenshot_2.jpg)
![Screenshot_3.jpg](imgs_sem_6%2FScreenshot_3.jpg)

Реузльтаты с lr = 0.005
![Screenshot_4.jpg](imgs_sem_6%2FScreenshot_4.jpg)
![Screenshot_5.jpg](imgs_sem_6%2FScreenshot_5.jpg)
![Screenshot_6.jpg](imgs_sem_6%2FScreenshot_6.jpg)

Реузльтаты с lr = 0.005 и episodes_to_train = 16
![Screenshot_7.jpg](imgs_sem_6%2FScreenshot_7.jpg)
![Screenshot_8.jpg](imgs_sem_6%2FScreenshot_8.jpg)
![Screenshot_9.jpg](imgs_sem_6%2FScreenshot_9.jpg)

в первом испытании среднее количество шагов, при котором алгоритм завершился, составил ~32133
во втором испытании уже среднее количество шагов составило ~35955
в третьем составило ~54047

Вывод: в рамках поставленного порогового значения остановки, стандартные параметры самые отптимальные,
однако в случае, если значение остановки будет в разы больше, полагаю, уже будут лучше показывать себя с другими значениями шиперпараметров, приблеженные к испытаниям 2 и 3.

# Задание 3
Для начала обучим модель на стандартных гиперпараметрах, получаем следующие показатели:
![Screenshot_task_3_1.jpg](imgs_sem_6%2FScreenshot_task_3_1.jpg)

![Screenshot_11.jpg](imgs_sem_6%2FScreenshot_11.jpg)
![Screenshot_12.jpg](imgs_sem_6%2FScreenshot_12.jpg)
![Screenshot_13.jpg](imgs_sem_6%2FScreenshot_13.jpg)
![Screenshot_14.jpg](imgs_sem_6%2FScreenshot_14.jpg)
![Screenshot_15.jpg](imgs_sem_6%2FScreenshot_15.jpg)
![Screenshot_16.jpg](imgs_sem_6%2FScreenshot_16.jpg)
![Screenshot_17.jpg](imgs_sem_6%2FScreenshot_17.jpg)
![Screenshot_18.jpg](imgs_sem_6%2FScreenshot_18.jpg)
![Screenshot_19.jpg](imgs_sem_6%2FScreenshot_19.jpg)
![Screenshot_20.jpg](imgs_sem_6%2FScreenshot_20.jpg)
![Screenshot_21.jpg](imgs_sem_6%2FScreenshot_21.jpg)
![Screenshot_22.jpg](imgs_sem_6%2FScreenshot_22.jpg)
![Screenshot_23.jpg](imgs_sem_6%2FScreenshot_23.jpg)

Разберемся, что означают все вышевыведенные величины:
"advantage": Разность между ожидаемыми и предсказанными значениями наград. Он используется для оценки того, насколько хорошо предсказываются будущие вознаграждения.
"values": Значения функции ценности состояния, предсказываемые моделью.
"batch_rewards": Фактические награды, полученные из пакета опыта.
"loss_entropy": Значение функции потерь, связанное с энтропией распределения вероятностей действий агента.
"loss_policy": Значение функции потерь, связанное с обновлением политики агента.
"loss_value": Значение функции потерь, связанное с обновлением функции ценности состояния.
"loss_total": Общее значение функции потерь, учитывающее потери энтропии, политики и значения.
"grad_l2": L2-норма всех градиентов модели.
"grad_max": Наибольшее абсолютное значение градиента модели.
"grad_var": Дисперсия всех градиентов модели.
Эти метрики используются для отслеживания процесса обучения и оценки производительности агента во время тренировки.

Изменять параметр мы будем NUM_ENVS.
num_envs - это количество сред (environments), которые используются для параллельного сбора опыта в методе Actor-Critic. В контексте данного кода, Actor-Critic метод использует несколько сред (environments) одновременно для сбора опыта. Каждая среда независимо взаимодействует с агентом в среде и собирает опыт (например, состояния, действия, вознаграждения) для обучения.
Уменьшим его в два раза, то есть опставим значение 25
Результаты следующие:
![Screenshot_31.jpg](imgs_sem_6%2FScreenshot_31.jpg)
![Screenshot_32.jpg](imgs_sem_6%2FScreenshot_32.jpg)
![Screenshot_33.jpg](imgs_sem_6%2FScreenshot_33.jpg)
![Screenshot_34.jpg](imgs_sem_6%2FScreenshot_34.jpg)
![Screenshot_35.jpg](imgs_sem_6%2FScreenshot_35.jpg)
![Screenshot_36.jpg](imgs_sem_6%2FScreenshot_36.jpg)
![Screenshot_37.jpg](imgs_sem_6%2FScreenshot_37.jpg)
![Screenshot_38.jpg](imgs_sem_6%2FScreenshot_38.jpg)
![Screenshot_39.jpg](imgs_sem_6%2FScreenshot_39.jpg)
![Screenshot_40.jpg](imgs_sem_6%2FScreenshot_40.jpg)
![Screenshot_41.jpg](imgs_sem_6%2FScreenshot_41.jpg)
![Screenshot_42.jpg](imgs_sem_6%2FScreenshot_42.jpg)
![Screenshot_43.jpg](imgs_sem_6%2FScreenshot_43.jpg)
![Screenshot_44.jpg](imgs_sem_6%2FScreenshot_44.jpg)

Мы видим, что количество итераций до сходимости уменьшилось в два раза, значит уменьшение количества сред до разумных размеров может привести к более быстрому проходу.
Повторим испытание:
![Screenshot_51.jpg](imgs_sem_6%2FScreenshot_51.jpg)
![Screenshot_52.jpg](imgs_sem_6%2FScreenshot_52.jpg)
![Screenshot_53.jpg](imgs_sem_6%2FScreenshot_53.jpg)
![Screenshot_54.jpg](imgs_sem_6%2FScreenshot_54.jpg)
![Screenshot_55.jpg](imgs_sem_6%2FScreenshot_55.jpg)
![Screenshot_56.jpg](imgs_sem_6%2FScreenshot_56.jpg)
![Screenshot_57.jpg](imgs_sem_6%2FScreenshot_57.jpg)
![Screenshot_58.jpg](imgs_sem_6%2FScreenshot_58.jpg)
![Screenshot_59.jpg](imgs_sem_6%2FScreenshot_59.jpg)
![Screenshot_60.jpg](imgs_sem_6%2FScreenshot_60.jpg)
![Screenshot_61.jpg](imgs_sem_6%2FScreenshot_61.jpg)
![Screenshot_62.jpg](imgs_sem_6%2FScreenshot_62.jpg)
![Screenshot_63.jpg](imgs_sem_6%2FScreenshot_63.jpg)
![Screenshot_64.jpg](imgs_sem_6%2FScreenshot_64.jpg)
Кол-во итераций до сходимости оказалоссь 10.6 миллионов, все еще почти на 4 миллиона меньше, чем при обучении на стандартных параметрах.
Можно предположить, что уменьшение на адекватное число num_envs может ускорить обучение.