---
title: "Стохастический градиентный спуск. Адаптивные методы. Оптимизация нейронных сетей"
author: Даня Меркулов, Петр Остроухов
institute: Оптимизация для всех! ЦУ
format: 
    beamer:
      pdf-engine: xelatex
      aspectratio: 169
      fontsize: 9pt
      section-titles: true
      incremental: true
      include-in-header: ../files/xeheader.tex  # Custom LaTeX commands and preamble
    beamer-cu-maga:
      pdf-engine: xelatex
      aspectratio: 169
      fontsize: 9pt
      section-titles: true
      incremental: true
      include-in-header: ../files/xeheader_cu.tex
      header-includes:
        - \newcommand{\cover}{../files/Методы вып_оптимизации_презентация_14.pdf}
header-includes:
 - \newcommand{\bgimage}{../files/back17.jpeg}
 - \usetikzlibrary{positioning, decorations.pathreplacing, shapes.misc, calc}
 - \newcommand{\tikznode}[2]{\tikz[baseline=(#1.base), remember picture] \node[inner sep=0pt] (#1) {$#2$};}
---


# Задача с конечной суммой

## Задача с конечной суммой

Рассмотрим задачу минимизации среднего значения функции на конечной выборке:
$$
\min_{x \in \mathbb{R}^p} f(x) = \min_{x \in \mathbb{R}^p}\frac{1}{n} \sum_{i=1}^n f_i(x)
$$
\pause
Шаг градиентного спуска для этой задачи:
$$
\tag{GD}
x_{k+1} = x_k - \frac{\alpha_k}{n} \sum_{i=1}^n \nabla f_i(x)
$$
\pause

* Сходимость с постоянным $\alpha$ или линейным поиском.
* Стоимость итерации линейна по $n$. Для ImageNet $n\approx 1.4 \cdot 10^7$, для WikiText $n \approx 10^8$. Для FineWeb $n \approx 15 \cdot 10^{12}$ токенов.

\pause

Перейдем от вычисления полного градиента к его несмещенной оценке. На каждой итерации будем выбирать индекс $i_k$ случайно и равномерно:
$$
\tag{SGD}
x_{k+1} = x_k - \alpha_k  \nabla f_{i_k}(x_k)
$$
При $p(i_k = i) = \frac{1}{n}$ стохастический градиент является несмещенной оценкой полного градиента:
$$
\mathbb{E}[\nabla f_{i_k}(x)] = \sum_{i=1}^{n} p(i_k = i) \nabla f_i(x) = \sum_{i=1}^{n} \frac{1}{n} \nabla f_i(x) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(x) = \nabla f(x)
$$
Это означает, что математическое ожидание стохастического градиента совпадает с истинным градиентом $f(x)$.


## Результаты для градиентного спуска

Стохастическая итерация в $n$ раз дешевле, но сколько шагов потребуется для достижения заданной точности?

. . .

Если $\nabla f$ липшицев, справедливы оценки:

\begin{center}
\begin{tabular}{c c c}
\toprule
Условие   & GD & SGD \\ 
\midrule
PL           & $\mathcal{O}\left(\log(1/\varepsilon)\right)$       & \uncover<2->{$\mathcal{O}\left(1/\varepsilon\right)$}          \\
Выпуклый       & $\mathcal{O}\left(1/\varepsilon\right)$             & \uncover<2->{$\mathcal{O}\left(1/\varepsilon^2\right)$}        \\
Невыпуклый   & $\mathcal{O}\left(1/\varepsilon\right)$             & \uncover<2->{$\mathcal{O}\left(1/\varepsilon^2\right)$}        \\
\bottomrule
\end{tabular}
\end{center}

. . .

* SGD имеет низкую стоимость итерации, но низкую скорость сходимости. 
  * Сублинейная скорость даже в сильно выпуклом случае.
  * Оценки скорости не могут быть улучшены при стандартных предположениях.
  * Оракул возвращает несмещенную аппроксимацию градиента с ограниченной дисперсией.

* Методы с ускорением и квазиньютоновские методы не улучшают асимптотическую скорость в стохастическом случае, влияя лишь на константы (узкое место — дисперсия, а не число обусловленности).

# Стохастический градиентный спуск (SGD)

## Типичное поведение

[!["Divergence"](sgd_lin_reg_divergence.jpeg){width=90%}](https://fmin.xyz/docs/visualizations/sgd_divergence.mp4)


## Гладкий PL-случай с постоянным шагом

:::{.callout-note appearance="simple"}
Пусть $f$ — $L$-гладкая функция, удовлетворяющая условию Поляка-Лоясиевича (PL) с константой $\mu>0$, а дисперсия стохастического градиента ограничена: $\mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2$. Тогда стохастический градиентный спуск с постоянным шагом $\alpha < \frac{1}{2\mu}$ гарантирует
$$
\mathbb{E}[f(x_{k})-f^{*}] \leq (1-2\alpha\mu)^{k} [f(x_{0})-f^{*}] + \frac{L\sigma^{2}\alpha}{4\mu}.
$$
:::

## Гладкий **выпуклый** случай

### Вспомогательные обозначения  
Для (возможно) неконстантной последовательности шагов $(\alpha_t)_{t\ge0}$ определим *взвешенное среднее*  
$$
\bar x_k \;\; \stackrel{\text{def}}{=}\;\; \frac{1}{\sum_{t=0}^{k-1}\alpha_t}\;\sum_{t=0}^{k-1}\alpha_t\,x_t\,,\qquad k\ge1 .
$$  
Везде ниже $f^{*}\equiv\min_x f(x)$ и $x^{*}\in\arg\min_x f(x)$.

---

## Гладкий выпуклый случай с **постоянным** шагом

:::{.callout-note appearance="simple"}
Пусть $f$ — выпуклая функция (не обязательно гладкая), а дисперсия стохастического градиента ограничена
$\mathbb{E}\!\left[\|\nabla f_{i_k}(x_k)\|^{2}\right]\le \sigma^{2}\;\;\forall k$. Если SGD использует постоянный шаг $\alpha_t\equiv\alpha > 0$, то для любого $k\ge1$
$$
\boxed{\;
\mathbb{E}\!\left[f(\bar x_k)-f^{*}\right]\;\le\;
\frac{\|x_0-x^{*}\|^{2}}{2\alpha\,k}\;+\;\frac{\alpha\,\sigma^{2}}{2}}
$$
где $\bar x_k = \frac{1}{k}\sum_{t=0}^{k-1} x_t$.

При выборе постоянного $\displaystyle \alpha=\frac{\|x_0-x^{*}\|}{\sigma\sqrt{k}}$ (зависящего от $k$) имеем
$$
\mathbb{E}\!\left[f(\bar x_k)-f^{*}\right]\le\frac{\|x_0-x^{*}\|\sigma}{\sqrt{k}}=\mathcal{O}\!\Bigl(\tfrac{1}{\sqrt{k}}\Bigr).
$$
:::

## Гладкий выпуклый случай с **убывающим** шагом  
$\displaystyle\alpha_k=\frac{\alpha_0}{\sqrt{k+1}},\quad 0<\alpha_0\le\frac{1}{4L}$

:::{.callout-note appearance="simple"}
При тех же предположениях, но с убывающим шагом $\alpha_k=\frac{\alpha_0}{\sqrt{k+1}}$
$$
\boxed{\;
\mathbb{E}\!\left[f(\bar x_k)-f^{*}\right]
\;\le\;
\frac{5\|x_0-x^{*}\|^{2}}{4\alpha_0\sqrt{k}}
\;+\;
5\alpha_0\sigma^{2}\,\frac{\log(k+1)}{\sqrt{k}}\;}
=\;\mathcal{O}\!\Bigl(\tfrac{\log k}{\sqrt{k}}\Bigr).
$$
:::

# Мини-батч SGD

## Мини-батч SGD

Детерминированный метод использует все $n$ градиентов:
$$
\nabla f(x_k) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(x_k).
$$

. . . 

Стохастический метод аппроксимирует это, используя только один элемент:
$$
\nabla f_{ik}(x_k) \approx \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(x_k).
$$

. . . 

Распространённый вариант — использовать выборку элементов $B_k$ («__мини-батч__»):
$$
\frac{1}{|B_k|} \sum_{i \in B_k} \nabla f_i(x_k) \approx \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(x_k),
$$
особенно полезно для векторизации и распараллеливания.

. . . 

Например, имея 16 ядер, можно взять $|B_k| = 16$ и вычислить 16 градиентов параллельно.

## Мини-батч как градиентный спуск с ошибкой

Метод SGD с батчем $B_k$ («мини-батч») использует итерации:
$$
x_{k+1} = x_k - \alpha_k \left(\frac{1}{|B_k|} \sum_{i \in B_k} \nabla f_i(x_k)\right).
$$

. . . 

Рассмотрим это как «градиентный метод с ошибкой»:
$$
x_{k+1} = x_k - \alpha_k(\nabla f(x_k) + e_k),
$$
где $e_k$ — разница между аппроксимированным и истинным градиентом.

. . . 

Если выбрать $\alpha_k = \frac{1}{L}$, то, согласно лемме о спуске:
$$
f(x_{k+1}) \leq f(x_k) - \frac{1}{2L} \|\nabla f(x_k)\|^2 + \frac{1}{2L} \|e_k\|^2,
$$
для любой ошибки $e_k$.

## Влияние ошибки на скорость сходимости

Оценка прогресса при $\alpha_k = \frac{1}{L}$ и ошибке градиента $e_k$:
$$
f(x_{k+1}) \leq f(x_k) - \frac{1}{2L} \|\nabla f(x_k)\|^2 + \frac{1}{2L} \|e_k\|^2.
$$

## Идея SGD и батчей

![](batches_1.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_2.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_3.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_4.pdf)

## Идея SGD и батчей {.noframenumbering}

![](batches_5.pdf)

## Основная проблема SGD

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](sgd_problems.pdf)

## Основные результаты сходимости SGD

:::{.callout-note appearance="simple"}
Пусть $f$ — $L$-гладкая $\mu$-сильно выпуклая функция, а дисперсия стохастического градиента ограничена ($\mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2$). Тогда траектория SGD с постоянным шагом $\alpha < \frac{1}{2\mu}$ будет гарантировать:
:::{.callout-note appearance="simple"}

$$
\mathbb{E}[f(x_{k+1}) - f^*] \leq (1 - 2\alpha \mu)^k[f(x_{0}) - f^*]  + \frac{L \sigma^2 \alpha }{ 4 \mu}.
$$
:::

. . .

:::{.callout-note appearance="simple"}
Пусть $f$ — $L$-гладкая $\mu$-сильно выпуклая функция, а дисперсия стохастического градиента ограничена ($\mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2$). Тогда SGD с убывающим шагом $\alpha_k = \frac{2k + 1 }{ 2\mu(k+1)^2}$ будет сходиться сублинейно:

$$
\mathbb{E}[f(x_{k+1}) - f^*] \leq \frac{L \sigma^2}{ 2 \mu^2 (k+1)}
$$
:::

## Summary

* SGD с постоянным шагом не сходится к оптимуму даже в PL (или сильно выпуклом) случае.
* SGD сходится сублинейно со скоростью $\mathcal{O}\left(\frac{1}{k}\right)$ для PL-случая. 
* Ускорение Нестерова/Поляка не улучшает скорость сходимости.
* Двухфазный метод Ньютона достигает $\mathcal{O}\left(\frac{1}{k}\right)$ без сильной выпуклости.

# Адаптивность или масштабирование

## Adagrad (Duchi, Hazan, and Singer 2010/Streeter and MacMahan 2010)

Популярный адаптивный метод. Обозначим $g^{(k)} = \nabla f_{i_k}(x^{(k-1)})$. Правило обновления для $j = 1, \dots, p$:

$$
v^{(k)}_j = v^{k-1}_j + (g_j^{(k)})^2
$$
$$
x_j^{(k)} = x_j^{(k-1)} - \alpha \frac{g_j^{(k)}}{\sqrt{v^{(k)}_j  + \epsilon}}
$$

. . . 

**Заметки:**

* AdaGrad не требует настройки шага обучения: $\alpha > 0$ — фиксированная константа, и шаг обучения автоматически уменьшается в ходе итераций.
* Шаг обучения для редких информативных признаков убывает медленно.
* Может существенно превосходить SGD на разреженных задачах.
* Основной недостаток — монотонное накопление квадратов градиентов в знаменателе. AdaDelta, Adam, AMSGrad и др. улучшают это, популярны в обучении глубоких нейронных сетей.
* Константа $\epsilon$ обычно устанавливается в $10^{-6}$ для предотвращения деления на ноль.

## RMSProp (Tieleman and Hinton, 2012)

Модификация AdaGrad, устраняющая проблему агрессивного монотонного убывания шага. Использует экспоненциальное скользящее среднее квадратов градиентов для настройки шага по каждой координате переменной. Пусть $g^{(k)} = \nabla f_{i_k}(x^{(k-1)})$ и правило обновления для $j = 1, \dots, p$:
$$
v^{(k)}_j = \gamma v^{(k-1)}_j + (1-\gamma) (g_j^{(k)})^2
$$
$$
x_j^{(k)} = x_j^{(k-1)} - \alpha \frac{g_j^{(k)}}{\sqrt{v^{(k)}_j + \epsilon}}
$$

. . . 

**Заметки:**

* RMSProp нормирует шаг обучения на корень из скользящего среднего квадратов градиентов.
* Обеспечивает более тонкую настройку шагов обучения, чем AdaGrad, что делает его подходящим для неизотропных задач.
* Широко используется при обучении нейронных сетей, особенно рекуррентных.

<!-- ## Adadelta (Zeiler, 2012)

Расширение RMSProp, нацеленное на снижение зависимости от вручную заданного глобального шага обучения. Вместо накопления всех прошлых квадратов градиентов Adadelta ограничивает окно накопленных прошлых градиентов фиксированным размером $w$. Механизм обновления не требует шага обучения $\alpha$:
$$
v^{(k)}_j = \gamma v^{(k-1)}_j + (1-\gamma) (g_j^{(k)})^2
$$
$$
\tilde{g}_j^{(k)} = \frac{\sqrt{{\Delta x_j^{(k-1)}} + \epsilon}}{\sqrt{v^{(k)}_j+ \epsilon}} g_j^{(k)}
$$
$$
x_j^{(k)} = x_j^{(k-1)} - \tilde{g}_j^{(k)}
$$
$$
\Delta x_j^{(k)} = \rho \Delta x_j^{(k-1)} + (1-\rho) (\tilde{g}_j^{(k)})^2
$$

**Заметки:**

* Adadelta адаптирует шаги обучения на основе скользящего окна обновлений градиентов, а не накопления всех прошлых градиентов. Таким образом, настроенные шаги обучения более устойчивы к изменениям динамики модели.
* Метод не требует начального установления шага обучения, что упрощает настройку.
* Часто используется в глубоком обучении, где масштабы параметров существенно различаются между слоями. -->

## Adam (Kingma and Ba, 2014) ^[[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)] ^[[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)]

Объединяет элементы из AdaGrad и RMSProp. Использует экспоненциальное скользящее среднее как градиентов, так и их квадратов.

::::{.columns}
:::{.column width="55%"}
\begin{tabular}{ll}
EMA: & $m_j^{(k)} = \beta_1 m_j^{(k-1)} + (1-\beta_1) g_j^{(k)}$ \\[1ex]
                           & $v_j^{(k)} = \beta_2 v_j^{(k-1)} + (1-\beta_2) \left(g_j^{(k)}\right)^2$ \\[1ex]
Коррекция смещения:         & $\hat{m}_j = \dfrac{m_j^{(k)}}{1-\beta_1^k}$ \\[1ex]
                           & $\hat{v}_j = \dfrac{v_j^{(k)}}{1-\beta_2^k}$ \\[1ex]
Обновление:                   & $x_j^{(k)} = x_j^{(k-1)} - \alpha\,\dfrac{\hat{m}_j}{\sqrt{\hat{v}_j} + \epsilon}$ \\
\end{tabular}
:::
:::{.column width="45%"}

. . .

**Заметки:**

* Компенсирует смещение к нулю на начальных итерациях, наблюдаемое в других методах (например, RMSProp), что делает оценки более точными.
* Одна из самых цитируемых научных работ в мире.
* В 2018-2019 годах вышли статьи, указывающие на ошибку в оригинальной статье
* Не сходится для некоторых простых задач (даже выпуклых)
* Почему-то очень хорошо работает для некоторых сложных задач
* Работает для языковых моделей значительно лучше, чем для задач компьютерного зрения. Почему?
:::
::::

## AdamW (Loshchilov & Hutter, 2017)

Решает проблему $\ell_2$-регуляризации в  Adam. Стандартная $\ell_2$-регуляризация добавляет $\lambda \|x\|^2$ к функции потерь, что дает добавку $\lambda x$ к градиенту. В Adam эта добавка масштабируется адаптивным шагом обучения $\left(\sqrt{\hat{v}_j} + \epsilon\right)$, связывая затухание весов (weight decay) с величиной шага. AdamW отделяет затухание весов от адаптации шага.

Правило обновления:
$$
m_j^{(k)} = \beta_1 m_j^{(k-1)} + (1-\beta_1) g_j^{(k)}
$$
$$
v_j^{(k)} = \beta_2 v_j^{(k-1)} + (1-\beta_2) (g_j^{(k)})^2
$$
$$
\hat{m}_j = \frac{m_j^{(k)}}{1-\beta_1^k}, \quad \hat{v}_j = \frac{v_j^{(k)} }{1-\beta_2^k}
$$
$$
x_j^{(k)} = x_j^{(k-1)} - \alpha \left( \frac{\hat{m}_j}{\sqrt{\hat{v}_j} + \epsilon} + \lambda x_j^{(k-1)} \right)
$$

. . .

**Заметки:**

* Слагаемое затухания весов $\lambda x_j^{(k-1)}$ добавляется *после* адаптивного шага по градиенту.
* Широко используется в обучении трансформаторов и других крупных моделей. Вариант по умолчанию для Hugging Face Trainer.

<!-- ## Shampoo (Gupta, Anil, et al., 2018; Anil et al., 2020)

**Заметки:**

* Цель — эффективнее учитывать информацию о кривизне, чем методы первого порядка.
* Вычислительно дороже, чем Adam, но может сходиться быстрее или приводить к лучшим решениям по числу шагов.
* Требует аккуратной реализации для эффективности (например, эффективного вычисления корней из обратной матрицы, обработки больших матриц).
* Существуют варианты для разных форм тензоров (например, для свёрточных слоёв). -->


## Много методов

[![](nag_gs.pdf){width=100%}](https://fmin.xyz/docs/visualizations/nag_gs.mp4){fig-align="center"}

# \faAtom \ Muon

## Новый подход к оптимизации ^[[KIMI K2: OPEN AGENTIC INTELLIGENCE](https://arxiv.org/pdf/2507.20534?)]

::::{.columns}
:::{.column width="50%"}
![](fig_MMLU_performance.pdf){width=100%, fig-align="center"}
:::
:::{.column width="50%"}
![](fig_GSM8k_performance.pdf){width=100%, fig-align="center"}
:::
::::

Модели, отмеченные звёздочкой, были обучены методом Muon, остальные модели были обучены другими алгоритмами оптимизации.

## Интуиция за методом Muon ^[[Презентация R. Gower](https://docs.google.com/presentation/d/1KDjkaIa-7UyjacQSsuU88GdZh_UJTl2jT188Qzfnlek)]

\begin{center}
\fontsize{18pt}{22pt}\selectfont
$$
\min_{x \in \mathbb{R}^p} \tikznode{func}{f(x)}
$$
\vspace{0.5cm}
$$
f(x) = \tikznode{linear}{f(\tikznode{wk}{x_k}) + \langle \nabla f(x_k), x - x_k \rangle} + \tikznode{bigo}{\mathcal{O}(\|x - x_k\|_2^2)}.
$$
\end{center}

\begin{tikzpicture}[remember picture, overlay]
    \tikzset{
        box/.style={draw=blue!50!black, rounded corners, very thick, align=center, fill=white, font=\small\sffamily},
        arrow/.style={->, >=latex, very thick, blue!50!black}
    }

    \pause
    % Loss Function
    \node[box, above right=0.2cm and 1.5cm of func] (loss_label) {Функция потерь};
    \draw[arrow] (loss_label) -- (func);

    \pause
    % Linear Approximation
    \draw[very thick, blue!50!black, decoration={brace, mirror, raise=5pt}, decorate] (linear.south west) -- (linear.south east) node[midway, below=15pt, box] (lin_label) {Линейная\\аппроксимация};

    \pause
    % Good approx
    \node[box, below=1cm of bigo] (bigo_label) {{Хорошее приближение}\\в окрестности $x_k$};
    \draw[arrow] (bigo_label) -- (bigo);

\end{tikzpicture}

## Интуиция за методом Muon. Градиентный спуск

\begin{center}
\fontsize{18pt}{22pt}\selectfont
\vspace{0.5cm}
$$
\begin{aligned}
x_{k+1} &= \argmin_{x \in \mathbb{R}^p} \left( f(x_k) + \langle \nabla f(x_k), x - x_k \rangle + \tikznode{prox}{\frac{1}{2\alpha} \|x - x_k\|_2^2} \right) \pause \\
&= x_k - \tikznode{lr}{\alpha} \nabla f(x_k)
\end{aligned}
$$
\end{center}

\begin{tikzpicture}[remember picture, overlay]
    \tikzset{
        box/.style={draw=blue!50!black, rounded corners, very thick, align=center, fill=white, font=\small\sffamily},
        arrow/.style={->, >=latex, very thick, blue!50!black}
    }

    \pause
    % Incentives to stay close
    \node[box, above right=0.5cm and -2.0cm of prox] (prox_label) {Штраф за\\дальность от $x_k$};
    \draw[arrow] (prox_label) -- (prox);

    \pause
    % Learning rate
    \node[box, below=1.0cm of lr] (lr_label) {Шаг обучения /\\коэффициент регуляризации};
    \draw[arrow] (lr_label) -- (lr);

    \node[anchor=south east, xshift=-0.3cm, yshift=0.5cm] at (current page.south east) {\includegraphics[width=0.2\paperwidth]{../files/muon_gd.jpeg}};

\end{tikzpicture}

## Интуиция за методом Muon. Нормированный градиентный спуск

\begin{center}
\fontsize{18pt}{22pt}\selectfont
\vspace{0.5cm}
$$
\begin{aligned}
x_{k+1} &= \argmin_{\tikznode{constr}{\|x - x_k\|_2 = \alpha}} \left( f(x_k) + \langle \nabla f(x_k), x - x_k \rangle \right) \pause \\
&= x_k - \tikznode{lr}{\alpha} \frac{\nabla f(x_k)}{\|\nabla f(x_k)\|_2}
\end{aligned}
$$
\end{center}

\begin{tikzpicture}[remember picture, overlay]
    \tikzset{
        box/.style={draw=blue!50!black, rounded corners, very thick, align=center, fill=white, font=\small\sffamily},
        arrow/.style={->, >=latex, very thick, blue!50!black}
    }

    \pause
    % Constraint
    \node[box, below left=0.1cm and -9.0cm of constr] (constr_label) {Ограничение на\\длину шага};
    \draw[arrow] (constr_label) -- (constr);

    \pause
    % Learning rate
    \node[box, below=1.0cm of lr] (lr_label) {Параметр ограничения /\\шаг обучения};
    \draw[arrow] (lr_label) -- (lr);

    \node[anchor=south east, xshift=-0.3cm, yshift=0.5cm] at (current page.south east) {\includegraphics[width=0.2\paperwidth]{../files/muon_lmo.jpeg}};

\end{tikzpicture}

## Что насчёт других норм?

![Примеры шаров в разных нормах](p_balls.pdf)

<!-- ## Для неевклидовых норм нужно ввести несколько определений

* Сопряжённая норма:
  $$
  \|g\|^* = \sup_{\|x\| = 1} \langle g, x \rangle
  $$
* Linear Minimization Oracle:
  $$
  \text{LMO}_{\|\cdot\|}(g) = \argmin_{\|x\| = 1} \langle g, x \rangle
  $$
* Важное свойство, связывающее эти два понятия:
  $$
  \langle g, \text{LMO}_{\|\cdot\|}(g) \rangle = -\|g\|^*
  $$ -->

## Неевклидовы записи методов ^[[Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)]

__Linear Minimization Oracle__:
  $$
  \text{LMO}_{\|\cdot\|}(g) = \argmin_{\|x\| = 1} \langle g, x \rangle
  $$

:::{.callout-important appearance="simple"}

### Неевклидов градиентный спуск

Для вектора градиента $g = \nabla f(x_k)$ и шага $\alpha > 0$:
$$
\begin{aligned}
x_{k+1} &= \argmin_{x \in \mathbb{R}^p} \left( f(x_k) + \langle g, x - x_k \rangle + \frac{1}{2\alpha} \|x - x_k\|^2 \right)
\end{aligned}
$$

<!-- 
$$
\begin{aligned}
x_{k+1} &= \argmin_{x \in \mathbb{R}^p} \left( f(x_k) + \langle g, x - x_k \rangle + \frac{1}{2\alpha} \|x - x_k\|^2 \right)\\
&= x_k + \alpha \|g\|^*\text{LMO}_{\|\cdot\|}(g)
\end{aligned}
$$
-->
:::

\pause

:::{.callout-important appearance="simple"}

### Неевклидов нормированный градиентный спуск

Для вектора градиента $g = \nabla f(x_k)$ и шага $\alpha > 0$:
$$
\begin{aligned}
x_{k+1} &= \argmin_{\|x - x_k\| = \alpha} \left( f(x_k) + \langle g, x - x_k \rangle \right)\\
&= x_k + \alpha \text{LMO}_{\|\cdot\|}(g)
\end{aligned}
$$

:::


## В нейросетях параметры — матрицы

* В линейных слоях, attention, embedding-слоях параметр — матрица весов
  $$
  W \in \mathbb{R}^{d \times n},\qquad
  G_k = \nabla_W f(W_k) \in \mathbb{R}^{d \times n}.
  $$
* Естественно использовать **матричные нормы**: операторную $\|\cdot\|_{\mathrm{op}}$, ядерную $\|\cdot\|_{\mathrm{nuc}}$, Фробениуса $\|\cdot\|_F$ и т.п.
* Вся логика переносится: вместо вектора ищем «лучшее направление спуска» среди матриц заданной длины.
* Cкалярное произведение:
  $$
  \langle A,B\rangle := \operatorname{tr}(A^\top B) = \sum_{ij} A_{ij}B_{ij}.
  $$

## Неевклидов нормированный спуск для матриц

Пусть заданы матричная норма $\|\cdot\|$ и шаг $\lambda>0$. Тогда нормированный шаг по матрице $W$:

$$
\begin{aligned}
W_{k+1}
&= \argmin_{\|W - W_k\| = \lambda}
\Bigl(
f(W_k) + \langle G_k, W - W_k \rangle
\Bigr) \
&= W_k + \lambda \text{LMO}_{\|\cdot\|}(G_k),
\end{aligned}
$$

где
$$
\text{LMO}_{\|\cdot\|}(G)
= \argmin_{\|W\|=1} \langle G, W\rangle
$$

— тот же самый LMO, только теперь он ищет **матрицу** единичной нормы, дающую наибольшее убывание линейного приближения.


## Операторная норма и быстрый расчёт ($UV^\top$)

Рассмотрим операторную (спектральную) норму $\|\cdot\|_{\mathrm{op}}$.
Пусть
$$
G_k = U\Sigma V^\top
$$
— редуцированное SVD градиента. Тогда

. . .

* LMO (с «max»-формулировкой) по операторной норме:
  $$
  \text{LMO}_{\|\cdot\|}(G)= - U V^\top,
  $$
  то есть оптимальное направление — **polar factor** (matrix sign) матрицы $G_k$.

* Проблема: полное SVD на каждом шаге дорого.
  Хорошая новость: нам нужен только ($UV^\top$), его можно считать гораздо быстрее:

* итерациями **Newton–Schulz**/ **Polar Express**, которые используют только матричные умножения, дают приближение $UV^\top$ за несколько шагов и снимают узкое место полного SVD внутри Muon.


## Обучение GPT-2 (124M) на FineWeb

![NanoGPT speedrun](nanogpt_speedrun81w.png){width=70%}

## Обучение GPT-2 (124M) на FineWeb

![NanoGPT speedrun](nanogpt_speedrun82w.png){width=70%}

# Оптимизация для глубокого обучения с практической точки зрения

## Как сравнивать методы? Бенчмарк AlgoPerf ^[[Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179)] ^[[Accelerating neural network training: An analysis of the AlgoPerf competition](https://openreview.net/forum?id=CtM5xjRSfm)]

* **Бенчмарк AlgoPerf:** Сравнивает алгоритмы обучения нейросетей в двух режимах:
    * **Внешняя настройка (*External Tuning*):** моделирует подбор гиперпараметров при ограниченных ресурсах (5 запусков, квазислучайный поиск). Оценка — медианное минимальное время достижения цели по 5 наборам задач.
    * **Самонастройка (*Self-Tuning*):** моделирует автоматический подбор на одной машине (фиксированный или внутренний подбор, бюджет ×3). Оценка — медианное время выполнения по 5 наборам задач.
* **Оценка:** результаты агрегируются с помощью профилей производительности. Профили показывают долю задач, решённых за время, не превышающее множитель $\tau$ относительно самой быстрой посылки. Итоговая оценка — нормированная площадь под кривой профиля (1.0 = самая быстрая на всех задачах).
* **Затраты ресурсов:** оценка требует $\sim 49,240$ часов суммарно на 8x NVIDIA V100 GPUs (в среднем $\sim 3469$ ч/внешняя настройка, $\sim 1847$ ч/самонастройка).

## Бенчмарк AlgoPerf

**Сводка _фиксированных_ базовых задач в бенчмарке AlgoPerf.** Функции потерь включают кросс‑энтропию (CE), среднюю абсолютную ошибку (L1) и функцию потерь CTC (Connectionist Temporal Classification). Дополнительные метрики оценки: индекс структурного сходства (SSIM), коэффициент ошибок (ER), доля ошибок по словам (WER), средняя усреднённая точность (mAP) и метрика BLEU (*bilingual evaluation understudy*). Бюджет времени выполнения соответствует правилам внешней настройки; правила самонастройки допускают обучение, в 3 раза более длительное.

| Задача                          | Датасет    | Модель                        | Функция потерь | Метрика | Целевое значение (валидация) | Бюджет времени |
|----------------------|----------------|----------------------|------|--------|--------------------|----------------|
| Clickthrough rate prediction | CRITEO 1TB | DLRM<sub>SMALL</sub>         | CE   | CE     | 0.123735           | 7703           |
| MRI reconstruction           | FASTMRI    | U-NET                        | L1   | SSIM   | 0.7344             | 8859           |
| Image classification         | IMAGENET   | ResNet-50                    | CE   | ER     | 0.22569            | 63,008         |
|                               |            | ViT                          | CE   | ER     | 0.22691            | 77,520         |
| Speech recognition           | LIBRISPEECH| Conformer                    | CTC  | WER    | 0.085884           | 61,068         |
|                               |            | DeepSpeech                   | CTC  | WER    | 0.119936           | 55,506         |
| Molecular property prediction| OGBG       | GNN                          | CE   | mAP    | 0.28098            | 18,477         |
| Translation                  | WMT        | Transformer                  | CE   | BLEU   | 30.8491            | 48,151         |


## Бенчмарк AlgoPerf

![](algoperf.png)


## Бенчмарк AlgoPerf

![](scores_max_tau_external_tuning.pdf){fig-align="center" width=86%}

## NanoGPT speedrun

![[\faLink\ Источник](https://github.com/KellerJordan/modded-nanogpt)](nanogpt_speedrun.pdf){width=96%}

## Работают ли трюки, если увеличить размер модели?

![[\faLink\ Источник](https://github.com/KellerJordan/modded-nanogpt/blob/master/img/nanogpt_speedrun51.png)](nanogpt_speedrun_scale.png){width=75%}

## Работают ли трюки, если увеличить размер модели?

![[\faLink\ Источник](https://github.com/KellerJordan/modded-nanogpt/blob/master/img/nanogpt_speedrun52.png)](nanogpt_speedrun_tokens.png){width=65%}

# Неожиданные истории

## Adam работает хуже для CV, чем для LLM? ^[[Linear attention is (maybe) all you need (to understand transformer optimization)](https://arxiv.org/abs/2310.01082)]

:::: {.columns}
::: {.column width="40%"}
![CNNs on MNIST and CIFAR10](cnns.pdf)
:::

::: {.column width="60%"}
![Transformers on PTB, WikiText2, and SQuAD](transformers.pdf)
:::
::::

Чёрные линии — SGD, красные — Adam.

## Почему Adam работает хуже для CV, чем для LLM? ^[[Linear attention is (maybe) all you need (to understand transformer optimization)](https://arxiv.org/abs/2310.01082)]

### Потому что шум градиентов в языковых моделях имеет тяжелые хвосты?

![](histogram_full.pdf)


## Почему Adam работает хуже для CV, чем для LLM? ^[[Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models](https://arxiv.org/abs/2402.19449)]

### Нет! Распределение меток имеет тяжёлые хвосты!

:::: {.columns}
::: {.column width="50%"}
В компьютерном зрении датасеты часто сбалансированы: 1000 котиков, 1000 песелей и т.д.

В языковых датасетах почти всегда не так: слово *the* встречается часто, слово *tie* — на порядки реже.
:::

::: {.column width="50%"}
![Распределение частоты токенов в PTB](PTB_classes.pdf){width=100%}
:::
::::

## Почему Adam работает хуже для CV, чем для LLM? ^[[Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models](https://arxiv.org/abs/2402.19449)]

### SGD медленно прогрессирует на редких классах

![](sgd_adam_imb.pdf){width=100% align="center"}
![](sgd_adam_imb_leg.pdf){width=100% align="center"}

SGD не добивается прогресса на низкочастотных классах, в то время как Adam добивается. Обучение GPT-2 S на WikiText-103. (a) Распределение классов, отсортированных по частоте встречаемости, разбитых на группы, соответствующие $\approx 10$ % данных. (b) Значение функции потерь при обучении. (c, d) Значение функции потерь при обучении для каждой группы при использовании SGD и Adam. 

## Влияние инициализации ^[[On the importance of initialization and momentum in deep learning Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton](https://proceedings.mlr.press/v28/sutskever13.html)]

:::{.callout-tip appearance="simple"}
Правильная инициализация нейронной сети важна. Функция потерь нейронной сети сильно невыпукла; оптимизировать её для достижения «хорошего» решения трудно, это требует тщательной настройки. 
:::

. . .

* Не инициализируйте все веса одинаково — почему?
* Случайная инициализация: инициализируйте случайно, например, из гауссовского распределения $N(0,\sigma^2)$, где стандартное отклонение $\sigma$ зависит от числа нейронов в слое. Это обеспечивает нарушение симметрии (*symmetry breaking*).
* Можно найти более полезные советы [здесь](https://cs231n.github.io/neural-networks-2/)

## Влияние инициализации весов нейронной сети на сходимость методов ^[[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1502.01852)]

:::: {.columns}
::: {.column width="50%"}
![22-layer ReLU net: good init converges faster](converge_22layers.pdf)
:::

::: {.column width="50%"}
![30-layer ReLU net: good init is able to converge](converge_30layers.pdf)
:::

::::

# Весёлые истории

## 

[![](gd_scalar_convergence.pdf)](https://fmin.xyz/docs/visualizations/sgd_3.mp4)

## 

[![](gd_scalar_convergence_to_local_minimum.pdf)](https://fmin.xyz/docs/visualizations/sgd_4.mp4)

## 

[![](sgd_escape.pdf)](https://fmin.xyz/docs/visualizations/sgd_5.mp4)


## Визуализация с помощью проекции на прямую

* Обозначим через $w_0$ начальные веса нейронной сети. Веса, полученные после обучения, обозначим $\hat{w}$.

. . .

* Сгенерируем случайное направление $w_1 \in \mathbb{R}^p$ той же размерности, затем вычислим значение функции потерь вдоль этого направления:

$$
L (\alpha) = L (w_0 + \alpha w_1), \quad \text{где } \alpha \in [-b, b].
$$

## Проекция функции потерь нейронной сети на прямую

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](../files/Line_projection_No Dropout.pdf)

## Проекция функции потерь нейронной сети на прямую {.noframenumbering}

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](../files/Line_projection_Dropout 0.pdf)

## Проекция функции потерь нейронной сети на плоскость

* Мы можем расширить эту идею и построить проекцию поверхности потерь на плоскость, которая задается 2 случайными векторами. 

\pause

* Два случайных гауссовых вектора в пространстве большой размерности с высокой вероятностью ортогональны. 

$$
L (\alpha, \beta) = L (w_0 + \alpha w_1 + \beta w_2), \quad \text{где } \alpha, \beta \in [-b, b]^2.
$$

\pause

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](plane_projection.jpeg){width=70%}

## Может ли быть полезно изучение таких проекций? ^[[Visualizing the Loss Landscape of Neural Nets, Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein](https://arxiv.org/abs/1712.09913)]

:::: {.columns}
::: {.column width="35%"}
![The loss surface of ResNet-56 without skip connections](noshortLog.png)
:::

::: {.column width="65%"}
![The loss surface of ResNet-56 with skip connections](shortHighResLog.png)
:::

::::

## Может ли быть полезно изучение таких проекций, если серьезно? ^[[Loss Landscape Sightseeing with Multi-Point Optimization, Ivan Skorokhodov, Mikhail Burtsev](https://arxiv.org/abs/1910.03867)]

![Examples of a loss landscape of a typical CNN model on FashionMNIST and CIFAR10 datasets found with MPO. Loss values are color-coded according to a logarithmic scale](icons-grid.png)


## Ширина локальных минимумов

![](sam_a.pdf)

## Ширина локальных минимумов{.noframenumbering}

![](sam_b.pdf)

## Ширина локальных минимумов{.noframenumbering}

![](sam_c.pdf)

## Экспоненциальный шаг обучения

* [Exponential Learning Rate Schedules for Deep Learning](http://www.offconvex.org/2020/04/24/ExpLR1/)

## Double Descent ^[[Reconciling modern machine learning practice and the bias-variance trade-off, Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal](https://arxiv.org/abs/1812.11118)]

![](doubledescent.pdf){width=100%}

## Double Descent

[![](dd.pdf)](https://fmin.xyz/docs/visualizations/double_descent.mp4)

## Grokking ^[[Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets,   Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra](https://arxiv.org/abs/2201.02177)]

:::: {.columns}
::: {.column width="50%"}
![Training transformer with 2 layers, width 128, and 4 attention heads, with a total of about $4 \cdot 10^5$ non-embedding parameters. Reproduction of experiments (\~ half an hour) is available [here](https://colab.research.google.com/drive/1r3Wg84XECq57fT2B1dvHLSJrJ2sjIDCJ?usp=sharing)](grokking.png)
:::

::: {.column width="50%"}

* Рекомендую посмотреть лекцию Дмитрия Ветрова **Удивительные свойства функции потерь в нейронной сети** (*Surprising properties of loss landscape in overparameterized models*). [\faYoutube \ видео](https://youtu.be/d60ShbSAu4A), [\faFile \ Презентация](https://disk.yandex.ru/i/OPtA2-8hSQRFNg)

* Автор [\faTelegram \ канала Свидетели Градиента](https://t.me/GradientWitnesses) собирает интересные наблюдения и эксперименты про гроккинг. 

* Также есть [\faYoutube \ видео](https://www.youtube.com/watch?v=pmHkDKPg0WM) с его докладом **Чем не является гроккинг**.

:::

::::

# Бонус: доказательства

## Следствие из липшицевости градиента

Из липшицевости градиента следует:
$$
f(x_{k+1}) \leq f(x_k) + \langle \nabla f(x_k), x_{k+1} - x_k \rangle + \frac{L}{2} \|x_{k+1}-x_k\|^2
$$ 

. . .

Используя $(\text{SGD})$:
$$
f(x_{k+1}) \leq f(x_k) - \alpha_k \langle \nabla f(x_k),  \nabla f_{i_k}(x_k)\rangle + \alpha_k^2\frac{L}{2} \|\nabla f_{i_k}(x_k)\|^2
$$

. . .

Теперь возьмем матожидание по $i_k$:
$$
\mathbb{E}[f(x_{k+1})] \leq \mathbb{E}[f(x_k) - \alpha_k \langle \nabla f(x_k),  \nabla f_{i_k}(x_k)\rangle + \alpha_k^2\frac{L}{2} \|\nabla f_{i_k}(x_k)\|^2]
$$

. . .

Используя линейность матожидания:
$$
\mathbb{E}[f(x_{k+1})] \leq f(x_k) - \alpha_k \langle \nabla f(x_k),  \mathbb{E}[\nabla f_{i_k}(x_k)]\rangle + \alpha_k^2\frac{L}{2} \mathbb{E}[\|\nabla f_{i_k}(x_k)\|^2]
$$

. . .

Так как выбор индекса равномерен, стохастический градиент несмещён: $\mathbb{E}[\nabla f_{i_k}(x_k)] = \nabla f(x_k)$:
$$
\mathbb{E}[f(x_{k+1})] \leq f(x_k) - \alpha_k \|\nabla f(x_k)\|^2 + \alpha_k^2\frac{L}{2} \mathbb{E}[\|\nabla f_{i_k}(x_k)\|^2]
$${#eq-sgd-decrement}

## Сходимость. Гладкий PL-случай

:::{.callout-note appearance="simple"}
Пусть $f$ — $L$-гладкая функция, удовлетворяющая условию Поляка-Лоясиевича (PL) с константой $\mu>0$, а дисперсия стохастического градиента ограничена: $\mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2$. Тогда стохастический градиентный спуск с постоянным шагом $\alpha < \frac{1}{2\mu}$ гарантирует
$$
\mathbb{E}[f(x_{k})-f^{*}] \leq (1-2\alpha\mu)^{k} [f(x_{0})-f^{*}] + \frac{L\sigma^{2}\alpha}{4\mu}.
$$
:::

Воспользуемся неравенством ([-@eq-sgd-decrement]):
$$
\begin{aligned}
\uncover<+->{ \mathbb{E}[f(x_{k+1})] &\leq f(x_k) - \alpha_k \|\nabla f(x_k)\|^2 + \alpha_k^2\frac{L}{2} \mathbb{E}[\|\nabla f_{i_k}(x_k)\|^2] \\ }
\uncover<+->{ \stackrel{\text{PL: } \|\nabla f(x_k)\|^2 \geq 2\mu(f(x_k) - f^*)}{ } \;\; } \uncover<+->{ &\leq f(x_k) - 2\alpha_k \mu (f(x_k) - f^*) + \alpha_k^2\frac{L}{2} \mathbb{E}[\|\nabla f_{i_k}(x_k)\|^2] \\ }
\uncover<+->{ {\stackrel{\text{Вычтем } f^*}{ }} \;\;} \uncover<+->{ \mathbb{E}[f(x_{k+1})] - f^* &\leq (f(x_k) - f^*) - 2\alpha_k \mu (f(x_k) - f^*) + \alpha_k^2\frac{L}{2} \mathbb{E}[\|\nabla f_{i_k}(x_k)\|^2] \\ }
\uncover<+->{ {\scriptsize \text{Переставляем}} \;\; &\leq (1 - 2\alpha_k \mu) [f(x_k) - f^*] + \alpha_k^2\frac{L}{2} \mathbb{E}[\|\nabla f_{i_k}(x_k)\|^2] \\ }
\uncover<+->{ \stackrel{\text{Ограниченность дисперсии: } \mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2}{ } \;\; } \uncover<+->{ &\leq (1 - 2\alpha_k \mu)[f(x_{k}) - f^*] + \frac{L \sigma^2 \alpha_k^2 }{2}. }
\end{aligned}
$$

## Сходимость. Гладкий PL-случай

:::{.callout-note appearance="simple"}
Пусть $f$ — $L$-гладкая функция, удовлетворяющая условию Поляка-Лоясиевича (PL) с константой $\mu>0$, а дисперсия стохастического градиента ограничена: $\mathbb{E}[\|\nabla f_i(x_k)\|^2] \leq \sigma^2$. Тогда стохастический градиентный спуск с убывающим шагом $\alpha_k = \frac{2k + 1 }{ 2\mu(k+1)^2}$ гарантирует
$$
\mathbb{E}[f(x_{k}) - f^*] \leq \frac{L \sigma^2}{ 2 \mu^2 k}
$$
:::

1. Рассмотрим стратегию **убывающего шага** с $\alpha_k = \frac{2k + 1 }{ 2\mu(k+1)^2}$. Получим:
  $$
  \begin{aligned}
  \uncover<+->{ \stackrel{1-2\alpha_k \mu = \frac{(k+1)^2}{(k+1)^2} - \frac{2k + 1 }{(k+1)^2} = \frac{k^2 }{ (k+1)^2}}{ }\;\;}\uncover<+->{ \mathbb{E}[f(x_{k+1}) - f^*] &\leq \frac{k^2 }{ (k+1)^2}[f(x_{k}) - f^*]  + \frac{L \sigma^2 (2k+1)^2}{ 8 \mu^2 (k+1)^4}} \\
  \uncover<+->{\stackrel{(2k+1)^2 < (2k + 2)^2 = 4(k+1)^2}{ } \;\; &\leq\frac{k^2 }{ (k+1)^2}[f(x_{k}) - f^*]  + \frac{L \sigma^2}{ 2 \mu^2 (k+1)^2}}
  \end{aligned}
  $$
2. Умножим обе части на $(k+1)^2$ и обозначим $\delta_f(k) \equiv k^2 \mathbb{E}[f(x_{k}) - f^*]$. Получим:
  $$
  \begin{aligned}
  (k+1)^2 \mathbb{E}[f(x_{k+1}) - f^*] &\leq k^2\mathbb{E}[f(x_{k}) - f^*]  + \frac{L\sigma^2 }{ 2 \mu^2} \\
  \delta_f(k+1) &\leq \delta_f(k)  + \frac{L\sigma^2 }{ 2 \mu^2}.
  \end{aligned}
  $$

## Сходимость. Гладкий PL-случай

3. Просуммируем предыдущее неравенство от $i=0$ до $k$ и, учитывая, что $\delta_f(0) = 0$, получим:
  $$
  \begin{aligned}
  \uncover<+->{\delta_f(i+1) &\leq \delta_f(i)  + \frac{L\sigma^2 }{ 2 \mu^2} \\ }
  \uncover<+->{\sum_{i=0}^k \left[ \delta_f(i+1) - \delta_f(i) \right] &\leq \sum_{i=0}^k \frac{L\sigma^2 }{ 2 \mu^2} \\ }
  \uncover<+->{\delta_f(k+1) - \delta_f(0)  &\leq \frac{L \sigma^2 (k+1)}{ 2 \mu^2} \\ }
  \uncover<+->{(k+1)^2 \mathbb{E}[f(x_{k+1}) - f^*] &\leq \frac{L \sigma^2 (k+1)}{ 2 \mu^2} \\ }
  \uncover<+->{\mathbb{E}[f(x_{k}) - f^*] &\leq \frac{L \sigma^2}{ 2 \mu^2 k}}
  \end{aligned}
  $$
  что даёт искомую скорость сходимости. 

## Сходимость. Гладкий выпуклый случай с **постоянным** шагом

:::{.callout-note appearance="simple"}
Пусть $f$ — выпуклая функция (не обязательно гладкая), а дисперсия стохастического градиента ограничена
$\mathbb{E}\!\left[\|\nabla f_{i_k}(x_k)\|^{2}\right]\le \sigma^{2}\;\;\forall k$. Если SGD использует постоянный шаг $\alpha_t\equiv\alpha > 0$, то для любого $k\ge1$
$$
\boxed{\;
\mathbb{E}\!\left[f(\bar x_k)-f^{*}\right]\;\le\;
\frac{\|x_0-x^{*}\|^{2}}{2\alpha\,k}\;+\;\frac{\alpha\,\sigma^{2}}{2}}
$$
где $\bar x_k = \frac{1}{k}\sum_{t=0}^{k-1} x_t$.

При выборе постоянного $\displaystyle \alpha=\frac{\|x_0-x^{*}\|}{\sigma\sqrt{k}}$ (зависящего от $k$) имеем
$$
\mathbb{E}\!\left[f(\bar x_k)-f^{*}\right]\le\frac{\|x_0-x^{*}\|\sigma}{\sqrt{k}}=\mathcal{O}\!\Bigl(\tfrac{1}{\sqrt{k}}\Bigr).
$$
:::

## Сходимость. Гладкий выпуклый случай с **постоянным** шагом

1.  Начнём с разложения квадрата расстояния до минимума:
    $$
    \|x_{k+1}-x^{*}\|^{2} = \|x_k - \alpha \nabla f_{i_k}(x_k) - x^*\|^2 =\|x_k-x^{*}\|^{2}-2\alpha\langle\nabla f_{i_k}(x_k),x_k-x^{*}\rangle
    +\alpha^{2}\|\nabla f_{i_k}(x_k)\|^{2}.
    $$

2.  Берём условное матожидание по $i_k$ (обозначим $\mathbb{E}_k[\cdot] = \mathbb{E}[\cdot | x_k]$), используем свойство $\mathbb{E}_k[\nabla f_{i_k}(x_k)] = \nabla f(x_k)$, ограниченность дисперсии $\mathbb{E}_k[\|\nabla f_{i_k}(x_k)\|^{2}] \le \sigma^2$ и выпуклость $f$ (которая даёт $\langle \nabla f(x_k), x_k - x^* \rangle \ge f(x_k) - f^*$):
    $$
    \begin{aligned}
    \mathbb{E}_k\!\bigl[\|x_{k+1}-x^{*}\|^{2}\bigr]
    &=\|x_k-x^{*}\|^{2}-2\alpha\langle\nabla f(x_k),x_k-x^{*}\rangle
    +\alpha^{2}\mathbb{E}_k\!\bigl[\|\nabla f_{i_k}(x_k)\|^{2}\bigr] \\
    &\le\|x_k-x^{*}\|^{2}-2\alpha\bigl(f(x_k)-f^{*}\bigr)
    +\alpha^{2}\sigma^{2}.
    \end{aligned}
    $$

3.  Переносим слагаемое с $f(x_k)$ влево и берём полное матожидание:
    $$
    2\alpha \mathbb{E}[f(x_k)-f^{*}] \le \mathbb{E}[\|x_k-x^{*}\|^{2}] - \mathbb{E}[\|x_{k+1}-x^{*}\|^{2}] + \alpha^{2}\sigma^{2}.
    $$

4.  Суммируем (телескопируем) по $t=0,\dots,k-1$:
    $$
    \begin{aligned}
    \sum_{t=0}^{k-1} 2\alpha\,\mathbb{E}\!\bigl[f(x_t)-f^{*}\bigr]
    &\le \sum_{t=0}^{k-1} \left( \mathbb{E}[\|x_t-x^{*}\|^{2}] - \mathbb{E}[\|x_{t+1}-x^{*}\|^{2}] \right) + \sum_{t=0}^{k-1} \alpha^{2}\sigma^{2} \\
    &= \mathbb{E}[\|x_0-x^{*}\|^{2}] - \mathbb{E}[\|x_k-x^{*}\|^{2}] + k\,\alpha^{2}\sigma^{2} \\
    &\le \|x_0-x^{*}\|^{2} + k\,\alpha^{2}\sigma^{2}.
    \end{aligned}
    $$

## Сходимость. Гладкий выпуклый случай с **постоянным** шагом

5.  Делим на $2\alpha k$:
    $$
    \frac{1}{k}\sum_{t=0}^{k-1}\mathbb{E}\!\bigl[f(x_t)-f^{*}\bigr] \le \frac{\|x_0-x^{*}\|^{2}}{2\alpha k} + \frac{\alpha \sigma^{2}}{2}.
    $$

6.  Воспользуемся выпуклостью $f$ и неравенством Йенсена для усреднённой точки $\bar x_k = \frac{1}{k}\sum_{t=0}^{k-1} x_t$:
    $$
    \mathbb{E}[f(\bar x_k)] \le \mathbb{E}\left[\frac{1}{k}\sum_{t=0}^{k-1} f(x_t)\right] = \frac{1}{k}\sum_{t=0}^{k-1} \mathbb{E}[f(x_t)].
    $$
    Вычитая $f^*$ из обеих частей, получаем:
    $$
    \mathbb{E}[f(\bar x_k) - f^*] \le \frac{1}{k}\sum_{t=0}^{k-1} \mathbb{E}[f(x_t) - f^*].
    $$

7.  Объединяя (5) и (6), получаем искомую оценку:
    $$
    \mathbb{E}\!\left[f(\bar x_k)-f^{*}\right]\;\le\;
    \frac{\|x_0-x^{*}\|^{2}}{2\alpha\,k}\;+\;\frac{\alpha\,\sigma^{2}}{2}.
    $$

