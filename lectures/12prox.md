---
title: "Проксимальный градиентный метод."
author: Даня Меркулов
institute: Оптимизация для всех! ЦУ
format: 
    beamer:
        pdf-engine: xelatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: true
        incremental: true
        include-in-header: ../files/xeheader.tex  # Custom LaTeX commands and preamble
header-includes:
 - \newcommand{\bgimage}{../files/back14.jpeg}
---

# Субградиентный метод

## Негладкие задачи

[![](l1_regularization.jpeg)](https://fmin.xyz/assets/Notebooks/Regularization_horizontal.mp4)

## Субградиентный метод

$$
\text{Субградиентный метод:} \qquad \qquad \min_{x \in \mathbb{R}^n} f(x) \qquad \qquad x_{k+1} = x_k - \alpha_k g_k, \quad g_k \in \partial f(x_k)
$$

. . .

|выпуклый (негладкий) | сильно выпуклый (негладкий) |
|:-----:|:-----:|
| $f(x_k) - f^* \sim  \mathcal{O} \left( \dfrac{1}{\sqrt{k}} \right)$ | $f(x_k) - f^* \sim  \mathcal{O} \left( \dfrac{1}{k} \right)$ | 
| $k_\varepsilon \sim  \mathcal{O} \left( \dfrac{1}{\varepsilon^2} \right)$ | $k_\varepsilon \sim \mathcal{O} \left( \dfrac{1}{\varepsilon} \right)$ | 

. . .

:::{.callout-theorem}

:::: {.columns}
::: {.column width="50%"}
Предположим, что $f$ является $G$-липшицевой и выпуклой, тогда субградиентный метод сходится как:
$$
f(\overline{x}) - f^* \leq \dfrac{G R}{ \sqrt{k}},
$$
:::

::: {.column width="50%"}
где

* $\alpha = \frac{R}{G\sqrt{k}}$
* $R = \|x_0 - x^*\|$
* $\overline{x} = \frac{1}{k}\sum\limits_{i=0}^{k-1} x_i$
:::
::::
:::

## Нижние оценки для негладких выпуклых задач

|выпуклый (негладкий) | сильно выпуклый (негладкий) |
|:-----:|:-----:|
| $f(x_k) - f^* \sim  \mathcal{O} \left( \dfrac{1}{\sqrt{k}} \right)$ | $f(x_k) - f^* \sim  \mathcal{O} \left( \dfrac{1}{k} \right)$ | 
| $k_\varepsilon \sim  \mathcal{O} \left( \dfrac{1}{\varepsilon^2} \right)$ | $k_\varepsilon \sim \mathcal{O} \left( \dfrac{1}{\varepsilon} \right)$ |

. . .

* Субградиентный метод является оптимальным для задач выше.
* Можно использовать метод зеркального спуска (обобщение метода субградиента на, возможно, неевклидову метрику) с той же скоростью сходимости, чтобы лучше согласовать геометрию задачи.
* Однако, мы можем достичь стандартной скорости градиентного спуска $\mathcal{O}\left(\frac1k \right)$ (и даже ускоренной версии $\mathcal{O}\left(\frac{1}{k^2} \right)$), если мы будем использовать структуру задачи.

# Проксимальный оператор

## Интуиция проксимального отображения

Рассмотрим дифференциальное уравнение градиентного потока:
$$
\dfrac{dx}{dt} = - \nabla f(x)
$$

:::: {.columns}
::: {.column width="50%"}
Явный метод Эйлера:

. . .

$$
\frac{x_{k+1} - x_k}{\alpha} = -\nabla f(x_k)
$$
Приводит к обычному методу градиентного спуска.
:::

. . .

::: {.column width="50%"}
Неявный метод Эйлера:
$$
\begin{aligned}
\uncover<+->{ \frac{x_{k+1} - x_k}{\alpha} = -\nabla f(x_{k+1}) \\ }
\uncover<+->{ \frac{x_{k+1} - x_k}{\alpha} + \nabla f(x_{k+1}) = 0 \\ }
\uncover<+->{ \left. \frac{x - x_k}{\alpha} + \nabla f(x)\right|_{x = x_{k+1}} = 0 \\ }
\uncover<+->{ \left. \nabla \left[ \frac{1}{2\alpha} \|x - x_k\|^2_2 + f(x) \right]\right|_{x = x_{k+1}} = 0 \\ }
\uncover<+->{ x_{k+1} = \text{arg}\min_{x\in \mathbb{R}^n} \left[ f(x) +  \frac{1}{2\alpha} \|x - x_k\|^2_2 \right] }
\end{aligned}
$$

:::
::::

. . .

:::{.callout-important}

### Проксимальный оператор

$$
\text{prox}_{f, \alpha}(x_k) = \text{arg}\min_{x\in \mathbb{R}^n} \left[ f(x) +  \frac{1}{2\alpha} \|x - x_k\|^2_2 \right]
$$
:::

## Визуализация проксимального оператора

![[Источник](https://twitter.com/gabrielpeyre/status/1273842947704999936)](prox_vis.jpeg){width=63%}

## Интуиция проксимального отображения

* **GD из метода проксимального отображения.** Возвращаемся к дискретизации:
    $$
    \begin{aligned}
    \uncover<+->{ x_{k+1} + \alpha \nabla f(x_{k+1}) &= x_k \\ }
    \uncover<+->{ (I + \alpha \nabla f ) (x_{k+1}) &= x_k \\ }
    \uncover<+->{ x_{k+1} = (I + \alpha \nabla f )^{-1} x_k &\stackrel{\alpha \to 0}{\approx} (I - \alpha \nabla f) x_k }
    \end{aligned}
    $$

    . . .

    Таким образом, мы получаем обычный градиентный спуск с $\alpha \to 0$: $x_{k+1} = x_k - \alpha \nabla f(x_k)$.
* **Метод Ньютона из метода проксимального отображения.** Теперь рассмотрим проксимальное отображение второго порядка приближения функции $f^{II}_{x_k}(x)$:
    $$
    \begin{aligned}
    \uncover<+->{ x_{k+1} = \text{prox}_{f^{II}_{x_k}, \alpha}(x_k) &=  \text{arg}\min_{x\in \mathbb{R}^n} \left[ f(x_k) + \langle \nabla f(x_k), x - x_k\rangle + \frac{1}{2} \langle \nabla^2 f(x_k)(x-x_k), x-x_k \rangle +  \frac{1}{2\alpha} \|x - x_k\|^2_2 \right] & \\ } 
    \uncover<+->{ & \left. \nabla f(x_{k}) + \nabla^2 f(x_k)(x - x_k) + \frac{1}{\alpha}(x - x_k)\right|_{x = x_{k+1}} = 0 & \\ }
    \uncover<+->{ & x_{k+1} = x_k - \left[\nabla^2 f(x_k) + \frac{1}{\alpha} I\right]^{-1} \nabla f(x_{k}) & }
    \end{aligned}
    $$

## От проекций к проксимальности

Пусть $\mathbb{I}_S$ — индикаторная функция для замкнутого, выпуклого множества $S$. Возвратимся к ортогональной проекции $\pi_S(y)$:

. . .

$$
\pi_S(y) := \arg\min_{x \in S} \frac{1}{2}\|x-y\|_2^2.
$$

. . .

С использованием следующего обозначения индикаторной функции
$$
\mathbb{I}_S(x) = \begin{cases} 0, &x \in S, \\ \infty, &x \notin S, \end{cases}
$$

. . .

Перепишем ортогональную проекцию $\pi_S(y)$ как
$$
\pi_S(y) := \arg\min_{x \in \mathbb{R}^n} \frac{1}{2} \|x - y\|^2 + \mathbb{I}_S (x).
$$

. . .

Проксимальность: заменим $\mathbb{I}_S$ на некоторую выпуклую функцию!
$$
\text{prox}_{r} (y) = \text{prox}_{r, 1} (y) := \arg\min \frac{1}{2} \|x - y\|^2 + r(x)
$$

# Составная оптимизация

## Регулярные / Составные целевые функции

:::: {.columns}
::: {.column width="50%"}
Многие негладкие задачи имеют вид
$$
\min_{x \in \mathbb{R}^n} \varphi(x) = f(x) + r(x)
$$

* **Lasso, L1-LS, compressed sensing** 
    $$
    f(x) = \frac12 \|Ax - b\|_2^2, r(x) = \lambda \|x\|_1
    $$
* **L1-логистическая регрессия, разреженная LR**
    $$
    f(x) = -y \log h(x) - (1-y)\log(1-h(x)), r(x) = \lambda \|x\|_1
    $$
:::

::: {.column width="50%"}
![](Composite_ru.pdf)
:::
::::

## Интуиция проксимального отображения

Условия оптимальности:
$$
\begin{aligned}
\uncover<+->{ 0 &\in \nabla f(x^*) + \partial r(x^*) \\ }
\uncover<+->{ 0 &\in \alpha \nabla f(x^*) + \alpha \partial r(x^*) \\ }
\uncover<+->{ x^* &\in \alpha \nabla f(x^*) + (I + \alpha \partial r)(x^*) \\ }
\uncover<+->{ x^* - \alpha \nabla f(x^*) &\in (I + \alpha \partial r)(x^*) \\ }
\uncover<+->{ x^* &= (I + \alpha \partial r)^{-1}(x^* - \alpha \nabla f(x^*)) \\ }
\uncover<+->{ x^* &= \text{prox}_{r, \alpha}(x^* - \alpha \nabla f(x^*)) }
\end{aligned}
$$

. . .

Которые приводят к методу проксимального градиента:
$$
x_{k+1} = \text{prox}_{r, \alpha}(x_k - \alpha \nabla f(x_k))
$$
И этот метод сходится со скоростью $\mathcal{O}(\frac{1}{k})$!

. . .

:::{.callout-note}
## Другая форма проксимального оператора
$$
\text{prox}_{f, \alpha}(x_k) = \text{prox}_{\alpha f}(x_k) = \text{arg}\min_{x\in \mathbb{R}^n} \left[ \alpha f(x) +  \frac{1}{2} \|x - x_k\|^2_2 \right] \qquad \text{prox}_{f}(x_k) = \text{arg}\min_{x\in \mathbb{R}^n} \left[ f(x) +  \frac{1}{2} \|x - x_k\|^2_2 \right] 
$$
:::

## Примеры проксимальных операторов

* $r(x) = \lambda \|x\|_1$, $\lambda > 0$
    $$
    [\text{prox}_r(x)]_i = \left[ |x_i| - \lambda \right]_+ \cdot \text{sign}(x_i),
    $$
    который также известен как оператор мягкого порога (soft-thresholding).
* $r(x) = \frac{\lambda}{2} \|x\|_2^2$, $\lambda > 0$
    $$
    \text{prox}_{r}(x) =  \frac{x}{1 + \lambda}.
    $$
* $r(x) = \mathbb{I}_S(x)$.
    $$
    \text{prox}_{r}(x_k - \alpha \nabla f(x_k)) = \text{proj}_{r}(x_k - \alpha \nabla f(x_k))
    $$

## Свойства проксимального оператора

:::{.callout-theorem}
Пусть $r: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ — выпуклая функция, для которой $\text{prox}_r$ определён. Если существует такой $\hat{x} \in \mathbb{R}^n$, что $r(x) < +\infty$, то проксимальный оператор определяется однозначно (т.е. всегда возвращает единственное значение).
:::
**Доказательство**: 

. . .

Проксимальный оператор возвращает минимум некоторой задачи оптимизации. 

. . .

Вопрос: Что можно сказать об этой задаче? 

. . .

Это сильно выпуклая функция, что означает, что она имеет единственный минимум (существование $\hat{x}$ необходимо для того, чтобы $r(\tilde{x}) + \frac{1}{2} \| x - \tilde{x} \|_2^2$ принимало конечное значение).

## Свойства проксимального оператора

:::{.callout-theorem}
Пусть $r : \mathbb{R}^n \rightarrow \mathbb{R} \cup \{+\infty\}$ — выпуклая функция, для которой $\text{prox}_r$ определен. Тогда, для любых $x, y \in \mathbb{R}^n$, следующие три условия эквивалентны:

* $\text{prox}_r(x) = y$,
* $x - y \in \partial r(y)$,
* $\langle x - y, z - y \rangle \leq r(z) - r(y)$ для любого $z \in \mathbb{R}^n$.

:::

**Доказательство**

:::: {.columns}
::: {.column width="50%"}
1. Установим эквивалентность между первым и вторым условиями. Первое условие можно переписать как
    $$
    y = \arg \min_{\tilde{x} \in \mathbb{R}^d} \left( r(\tilde{x}) + \frac{1}{2} \| x - \tilde{x} \|^2 \right).
    $$
    Из условий оптимальности для выпуклой функции $r$, это эквивалентно:
    $$
    0 \in \left.\partial \left( r(\tilde{x}) + \frac{1}{2} \| x - \tilde{x} \|^2 \right)\right|_{\tilde{x} = y} = \partial r(y) + y - x.
    $$

:::

::: {.column width="50%"}
2. Из определения субдифференциала, для любого субградиента $g \in \partial f(y)$ и для любого $z \in \mathbb{R}^d$:
    $$
    \langle g, z - y \rangle \leq r(z) - r(y).
    $$
    В частности, это верно для $g = x - y$. Обратно это также очевидно: для $g = x - y$, вышеуказанное соотношение выполняется, что означает $g \in \partial r(y)$.
:::
::::


## Свойства проксимального оператора

:::{.callout-theorem}
Оператор $\text{prox}_{r}(x)$ является жёстко нерастягивающим (FNE):
$$
\|\text{prox}_{r}(x) -\text{prox}_{r}(y)\|_2^2 \leq \langle\text{prox}_{r}(x)-\text{prox}_{r}(y), x-y\rangle
$$
и нерастягивающим:
$$
\|\text{prox}_{r}(x) -\text{prox}_{r}(y)\|_2 \leq \|x-y \|_2
$$
:::

**Доказательство**

:::: {.columns}
::: {.column width="50%"}
1. Пусть $u = \text{prox}_r(x)$, и $v = \text{prox}_r(y)$. Тогда, из предыдущего свойства:
    $$
    \begin{aligned}
    \langle x - u, z_1 - u \rangle \leq r(z_1) - r(u) \\
    \langle y - v, z_2 - v \rangle \leq r(z_2) - r(v).
    \end{aligned}
    $$

2. Заменим $z_1 = v$ и $z_2 = u$ и сложим:
    $$
    \begin{aligned}
    \langle x - u, v - u \rangle + \langle y - v, u - v \rangle \leq 0,\\
    \langle x - y, v - u \rangle + \|v - u\|^2_2 \leq 0.
    \end{aligned}
    $$
:::

::: {.column width="50%"}
3. Что и требовалось доказать после подстановки $u$ и $v$. 
    $$
    \|u -v\|_2^2 \leq \langle x - y, u - v \rangle 
    $$

4. Последнийпункт следует из неравенства Коши-Буняковского для последнего неравенства.
:::
::::

## Свойства проксимального оператора

:::{.callout-theorem}
Пусть $f: \mathbb{R}^n \rightarrow \mathbb{R} \cup \{+\infty\}$ и $r: \mathbb{R}^n \rightarrow \mathbb{R} \cup \{+\infty\}$ — выпуклые функции. Кроме того, пусть $f$ непрерывно дифференцируема и $L$-гладкая, а для $r$, $\text{prox}_r$ определена. Тогда, $x^*$ является решением составной задачи оптимизации тогда и только тогда, когда для любого $\alpha > 0$, выполняется:
$$
x^* = \text{prox}_{r, \alpha}(x^* - \alpha \nabla f(x^*))
$$
:::

**Доказательство**

1. Условия оптимальности:
    $$
    \begin{aligned}
    \uncover<+->{ 0 \in & \nabla f(x^*) + \partial r(x^*) \\ }
    \uncover<+->{ - \alpha \nabla f(x^*) \in & \alpha \partial r(x^*) \\ }
    \uncover<+->{ x^* - \alpha \nabla f(x^*) - x^* \in & \alpha \partial r(x^*) }
    \end{aligned}
    $$
2. Возвратимся к предыдущей лемме: 
    $$
    \text{prox}_r(x) = y \Leftrightarrow x - y \in \partial r(y)
    $$
3. Наконец, 
    $$
    x^* = \text{prox}_{\alpha r}(x^* - \alpha \nabla f(x^*)) = \text{prox}_{r, \alpha}(x^* - \alpha \nabla f(x^*))
    $$



# Теоретические инструменты для анализа сходимости

## Анализ сходимости \faGem \ \faGem[regular] \faGem[regular] \faGem[regular]

:::{.callout-theorem}
Пусть $f: \mathbb{R}^n \rightarrow \mathbb{R}$ — $L$-гладкая выпуклая функция. Тогда, для любых $x, y \in \mathbb{R}^n$, выполняется неравенство:  
$$
\begin{aligned}
f(x) + \langle \nabla f(x), y - x \rangle + \frac{1}{2L} & \|\nabla f(x) - \nabla f(y)\|^2_2 \leq f(y) \text{ или, эквивалентно, }\\
\|\nabla f(y)-\nabla f (x)\|_2^2 = & \|\nabla f(x)-\nabla f (y)\|_2^2 \leq 2L\left(f(x)-f(y)-\langle\nabla f (y),x -y\rangle \right)
\end{aligned}
$$
:::

**Доказательство.**

1. Рассмотрим другую функцию $\varphi(y) = f(y) - \langle \nabla f(x), y\rangle$. Очевидно, это выпуклая функция (как сумма выпуклых функций). И легко проверить, что она является $L$-гладкой функцией по определению, так как $\nabla \varphi(y) = \nabla f(y) - \nabla f(x)$ и $\|\nabla \varphi(y_1) - \nabla \varphi(y_2)\| = \|\nabla f(y_1) - \nabla f(y_2)\| \leq L\|y_1 - y_2\|$.
2. Теперь рассмотрим свойство гладкости параболы для функции $\varphi(y)$:
  $$
  \begin{aligned}
  \uncover<+->{ \varphi(y) & \leq  \varphi(x) + \langle \nabla \varphi(x), y-x \rangle + \frac{L}{2}\|y-x\|_2^2 \\ }
  \uncover<+->{ \stackrel{x := y, y := y - \frac1L \nabla\varphi(y)}{ }\;\;\varphi\left(y - \frac1L \nabla\varphi(y)\right) &  \leq \varphi(y) + \left\langle \nabla \varphi(y), - \frac1L \nabla\varphi(y)\right\rangle + \frac{1}{2L}\|\nabla\varphi(y)\|_2^2 \\ }
  \uncover<+->{ \varphi\left(y - \frac1L \nabla\varphi(y)\right) &  \leq \varphi(y) - \frac{1}{2L}\|\nabla\varphi(y)\|_2^2 }
  \end{aligned}
  $$

## Анализ сходимости \faGem \ \faGem \ \faGem[regular] \faGem[regular]

3. Из условий первого порядка для выпуклой функции $\nabla \varphi (y) =\nabla f(y) - \nabla f(x) = 0$, мы можем заключить, что для любого $x$, минимум функции $\varphi(y)$ находится в точке $y=x$. Следовательно:
  $$
  \varphi(x) \leq \varphi\left(y - \frac1L \nabla\varphi(y)\right) \leq \varphi(y) - \frac{1}{2L}\|\nabla\varphi(y)\|_2^2
  $$
4. Теперь, подставим $\varphi(y) = f(y) - \langle \nabla f(x), y\rangle$:
  $$
  \begin{aligned}
  \uncover<+->{ & f(x) - \langle \nabla f(x), x\rangle \leq f(y) - \langle \nabla f(x), y\rangle - \frac{1}{2L}\|\nabla f(y) - \nabla f(x)\|_2^2 \\ }
  \uncover<+->{ & f(x) + \langle \nabla f(x), y - x \rangle + \frac{1}{2L} \|\nabla f(x) - \nabla f(y)\|^2_2 \leq f(y) \\ }
  \uncover<+->{ & \|\nabla f(y) - \nabla f(x)\|^2_2 \leq 2L \left( f(y) - f(x) - \langle \nabla f(x), y - x \rangle \right) \\ }
  \uncover<+->{ {\scriptsize \text{поменять местами x и y}} \quad & \|\nabla f(x)-\nabla f (y)\|_2^2 \leq 2L\left(f(x)-f(y)-\langle\nabla f (y),x -y\rangle \right)}
  \end{aligned}
  $$

. . .

Лемма доказана. С первого взгляда она не имеет большого геометрического смысла, но мы будем использовать ее как удобный инструмент для оценки разницы между градиентами.

## Анализ сходимости \faGem \ \faGem \ \faGem \ \faGem[regular]

:::{.callout-theorem}
Пусть $f: \mathbb{R}^n \rightarrow \mathbb{R}$ непрерывно дифференцируема на $\mathbb{R}^n$. Тогда, функция $f$ является $\mu$-сильно выпуклой тогда и только тогда, когда для любых $x, y \in \mathbb{R}^d$ выполняется следующее:
$$
\begin{aligned}
\text{Strongly convex case } \mu >0 & &\langle \nabla f(x) - \nabla f(y), x - y \rangle &\geq \mu \|x - y\|^2 \\
\text{Convex case } \mu = 0 & &\langle \nabla f(x) - \nabla f(y), x - y \rangle &\geq 0
\end{aligned}
$$
:::

**Доказательство**

1. Мы докажем только случай сильной выпуклости, случай выпуклости следует из него с установкой $\mu=0$. Начнем с необходимости. Для сильно выпуклой функции
  $$
  \begin{aligned}
  & f(y) \geq f(x) + \langle \nabla f(x), y-x\rangle + \frac{\mu}{2}\|x-y\|_2^2 \\
  & f(x) \geq f(y) + \langle \nabla f(y), x-y\rangle + \frac{\mu}{2}\|x-y\|_2^2 \\
  {\scriptsize \text{sum}} \;\; & \langle \nabla f(x) - \nabla f(y), x - y \rangle \geq \mu \|x - y\|^2
  \end{aligned}
  $$

## Анализ сходимости \faGem \ \faGem \ \faGem \ \faGem

2. Для достаточности мы предполагаем, что $\langle \nabla f(x) - \nabla f(y), x - y \rangle \geq \mu \|x - y\|^2$. Используя теорему Ньютона-Лейбница $f(x) = f(y) + \int_{0}^{1} \langle \nabla f(y + t(x - y)), x - y \rangle dt$:
  $$
  \begin{aligned}
  \uncover<+->{ f(x) - f(y) - \langle \nabla f(y), x - y \rangle &= \int_{0}^{1} \langle \nabla f(y + t(x - y)), x - y \rangle dt - \langle \nabla f(y), x - y \rangle \\ }
  \uncover<+->{ \stackrel{ \langle \nabla f(y), x - y \rangle = \int_{0}^{1}\langle \nabla f(y), x - y \rangle dt}{ }\qquad &= \int_{0}^{1} \langle \nabla f(y + t(x - y)) - \nabla f(y), (x - y) \rangle dt \\ }
  \uncover<+->{ \stackrel{ y + t(x - y) - y = t(x - y)}{ }\qquad&= \int_{0}^{1} t^{-1} \langle \nabla f(y + t(x - y)) - \nabla f(y), t(x - y) \rangle dt \\ }
  \uncover<+->{ & \geq \int_{0}^{1} t^{-1} \mu \| t(x - y) \|^2 dt } \uncover<+->{ = \mu \| x - y \|^2 \int_{0}^{1} t dt} \uncover<+->{= \frac{\mu}{2} \| x - y \|^2_2 }
  \end{aligned}
  $$

  . . .

  Таким образом, мы получаем критерий сильной выпуклости, удовлетворяющий
  $$
  \begin{aligned}
  \uncover<+->{ & f(x) \geq f(y) + \langle \nabla f(y), x - y \rangle + \frac{\mu}{2} \| x - y \|^2_2} \uncover<+->{ \text{ или, эквивалентно: }\\ }
  \uncover<+->{ {\scriptsize \text{поменять местами x и y}} \quad & - \langle \nabla f(x), x - y \rangle \leq - \left(f(x) - f(y) + \frac{\mu}{2} \| x - y \|^2_2 \right) }
  \end{aligned}
  $$

# Проксимальный метод градиента. Выпуклый случай

## Сходимость

:::{.callout-theorem}

Рассмотрим проксимальный метод градиента
$$
x_{k+1} = \text{prox}_{\alpha r}\left(x_k - \alpha \nabla f(x_k)\right)
$$
Для критерия $\varphi(x) = f(x) + r(x)$, мы предполагаем:

::: {.nonincremental}
* $f$ выпукла, дифференцируема, $\text{dom}(f) = \mathbb{R}^n$, и $\nabla f$ является липшицевой с константой $L > 0$.
* $r$ выпукла, и $\text{prox}_{\alpha r}(x_k) = \text{arg}\min\limits_{x\in \mathbb{R}^n} \left[ \alpha r(x) +  \frac{1}{2} \|x - x_k\|^2_2 \right]$ может быть вычислен.
:::

Проксимальный градиентный спуск с фиксированным шагом $\alpha = 1/L$ удовлетворяет
$$
\varphi(x_k) - \varphi^* \leq \frac{L \|x_0 - x^*\|^2}{2 k},
$$
:::

Проксимальный градиентный спуск имеет скорость сходимости $O(1/k)$ или $O(1/\varepsilon)$. Это соответствует скорости градиентного спуска! (Но помните о стоимости проксимальной операции)

## Анализ сходимости \faGem \ \faGem[regular] \faGem[regular] \faGem[regular] \faGem[regular]

**Доказательство**

1. Введем **градиентное отображение**, обозначаемое как $G_{\alpha}(x)$, действующее как "градиентный объект":
  $$
  \begin{aligned}
  x_{k+1} &= \text{prox}_{\alpha r}(x_k - \alpha \nabla f(x_k))\\
  x_{k+1} &= x_k - \alpha G_{\alpha}(x_k).
  \end{aligned}
  $$
  где $G_{\alpha}(x)$ является:
  $$
  G_{\alpha}(x) = \frac{1}{\alpha} \left( x - \text{prox}_{\alpha r}\left(x - \alpha \nabla f\left(x\right)\right) \right)
  $$
  Очевидно, что $G_{\alpha}(x) = 0$ тогда и только тогда, когда $x$ оптимален. Следовательно, $G_{\alpha}$ аналогичен $\nabla f$. Если $x$ локально оптимален, то $G_{\alpha}(x) = 0$ даже для невыпуклой $f$. Это демонстрирует, что проксимальный градиентный метод эффективно объединяет градиентный спуск на $f$ с проксимальным оператором $r$, позволяя ему эффективно обрабатывать недифференцируемые компоненты.

2. Мы будем использовать гладкость и выпуклость $f$ для некоторой произвольной точки $x$:
  $$
  \begin{aligned}
  \uncover<+->{ {\scriptsize \text{гладкость}} \;\; f(x_{k+1}) &\leq  f(x_k) + \langle \nabla f(x_k), x_{k+1}-x_k \rangle + \frac{L}{2}\|x_{k+1}-x_k\|_2^2 \\ }
  \uncover<+->{ \stackrel{\text{выпуклость } f(x) \geq f(x_k) + \langle \nabla f(x_k), x-x_k \rangle}{ } \;\; } \uncover<+->{ &\leq f(x) - \langle \nabla f(x_k), x-x_k \rangle + \langle \nabla f(x_k), x_{k+1}-x_k \rangle + \frac{\alpha^2 L}{2}\|G_{\alpha}(x_k)\|_2^2 \\ }
  \uncover<+->{ &\leq f(x) + \langle \nabla f(x_k), x_{k+1}-x \rangle + \frac{\alpha^2 L}{2}\|G_{\alpha}(x_k)\|_2^2 }
  \end{aligned}
  $$

## Анализ сходимости \faGem \ \faGem \ \faGem[regular] \faGem[regular] \faGem[regular]

3. Теперь мы будем использовать свойство проксимального оператора, которое было доказано ранее:
  $$
  \begin{aligned}
  \uncover<+->{ x_{k+1} = \text{prox}_{\alpha r}\left(x_k - \alpha \nabla f(x_k)\right) \qquad  \Leftrightarrow \qquad x_k - \alpha \nabla f(x_k) - x_{k+1} \in \partial \alpha r (x_{k+1}) \\ }
  \uncover<+->{ \text{Так как } x_{k+1} - x_k = - \alpha G_{\alpha}(x_k) \qquad \Rightarrow \qquad  \alpha G_{\alpha}(x_k) - \alpha \nabla f(x_k) \in \partial \alpha r (x_{k+1}) \\ }
  \uncover<+->{ G_{\alpha}(x_k) - \nabla f(x_k) \in \partial r (x_{k+1}) }
  \end{aligned}
  $$
4. Из определения субградиента выпуклой функции $r$ для любой точки $x$:
  $$
  \begin{aligned}
  \uncover<+->{ & r(x) \geq r(x_{k+1}) + \langle g, x - x_{k+1} \rangle, \quad g \in \partial r (x_{k+1}) \\ }
  \uncover<+->{ {\scriptsize \text{подставить конкретный субградиент}} \qquad & r(x) \geq r(x_{k+1}) + \langle G_{\alpha}(x_k) - \nabla f(x), x - x_{k+1} \rangle \\ }
  \uncover<+->{ & r(x) \geq r(x_{k+1}) + \langle G_{\alpha}(x_k), x - x_{k+1} \rangle - \langle \nabla f(x), x - x_{k+1} \rangle \\ }
  \uncover<+->{ & \langle \nabla f(x),x_{k+1} - x \rangle \leq r(x) - r(x_{k+1}) - \langle G_{\alpha}(x_k), x - x_{k+1} \rangle }
  \end{aligned}
  $$
5. Учитывая приведённую выше оценку, мы возвращаемся к гладкости и выпуклости:
  $$
  \begin{aligned}
  \uncover<+->{ f(x_{k+1}) &\leq f(x) + \langle \nabla f(x_k), x_{k+1}-x \rangle + \frac{\alpha^2 L}{2}\|G_{\alpha}(x_k)\|_2^2 \\ }
  \uncover<+->{ f(x_{k+1}) &\leq f(x) + r(x) - r(x_{k+1}) - \langle G_{\alpha}(x_k), x - x_{k+1} \rangle + \frac{\alpha^2 L}{2}\|G_{\alpha}(x_k)\|_2^2 \\ }
  \uncover<+->{ f(x_{k+1}) + r(x_{k+1}) &\leq f(x) + r(x) - \langle G_{\alpha}(x_k), x - x_k + \alpha G_{\alpha}(x_k) \rangle + \frac{\alpha^2 L}{2}\|G_{\alpha}(x_k)\|_2^2 }
  \end{aligned}
  $$

## Анализ сходимости \faGem \ \faGem \ \faGem \ \faGem[regular] \faGem[regular]

6. Используя $\varphi(x) = f(x) + r(x)$ мы можем доказать очень полезное неравенство, которое позволит нам продемонстрировать монотонное убывание итерации:
  $$
  \begin{aligned}
  \uncover<+->{ & \varphi(x_{k+1}) \leq \varphi(x) - \langle G_{\alpha}(x_k), x - x_k \rangle - \langle G_{\alpha}(x_k), \alpha G_{\alpha}(x_k) \rangle + \frac{\alpha^2 L}{2}\|G_{\alpha}(x_k)\|_2^2 \\ }
  \uncover<+->{ & \varphi(x_{k+1}) \leq \varphi(x) + \langle G_{\alpha}(x_k), x_k - x \rangle + \frac{\alpha}{2} \left( \alpha L - 2 \right) \|G_{\alpha}(x_k) \|_2^2 \\ }
  \uncover<+->{ \stackrel{\alpha \leq \frac1L \Rightarrow \frac{\alpha}{2} \left( \alpha L - 2 \right) \leq  -\frac{\alpha}{2}}{ } \quad } \uncover<+->{ & \varphi(x_{k+1}) \leq \varphi(x) + \langle G_{\alpha}(x_k), x_k - x \rangle - \frac{\alpha}{2} \|G_{\alpha}(x_k) \|_2^2 }
  \end{aligned}
  $$
7. Теперь легко проверить, что когда $x = x_k$ мы получаем монотонное убывание для проксимального градиентного метода:
  $$
  \varphi(x_{k+1}) \leq \varphi(x_k) - \frac{\alpha}{2} \|G_{\alpha}(x_k) \|_2^2
  $$

## Анализ сходимости \faGem \ \faGem \ \faGem \ \faGem \ \faGem[regular]

8. Когда $x = x^*$:
  $$
  \begin{aligned}
  \uncover<+->{ \varphi(x_{k+1}) &\leq \varphi(x^*) + \langle G_{\alpha}(x_k), x_k - x^* \rangle - \frac{\alpha}{2} \|G_{\alpha}(x_k) \|_2^2 \\ }
  \uncover<+->{ \varphi(x_{k+1}) - \varphi(x^*) &\leq \langle G_{\alpha}(x_k), x_k - x^* \rangle - \frac{\alpha}{2} \|G_{\alpha}(x_k) \|_2^2 \\ } 
  \uncover<+->{ &\leq \frac{1}{2\alpha}\left[2 \langle \alpha G_{\alpha}(x_k), x_k - x^* \rangle - \|\alpha G_{\alpha}(x_k) \|_2^2\right] \\ }
  \uncover<+->{ &\leq \frac{1}{2\alpha}\left[2 \langle \alpha G_{\alpha}(x_k), x_k - x^* \rangle - \|\alpha G_{\alpha}(x_k) \|_2^2 - \|x_k - x^* \|_2^2 + \|x_k - x^* \|_2^2\right] \\ }
  \uncover<+->{ &\leq \frac{1}{2\alpha}\left[- \|x_k - x^* -  \alpha G_{\alpha}(x_k)\|_2^2 + \|x_k - x^* \|_2^2\right] \\ }
  \uncover<+->{ &\leq \frac{1}{2\alpha}\left[\|x_k - x^* \|_2^2 - \|x_{k+1} - x^* \|_2^2\right] }
  \end{aligned}
  $$

## Анализ сходимости \faGem \ \faGem \ \faGem \ \faGem \ \faGem

9. Теперь мы запишем приведенное выше ограничение для всех итераций $i \in 0, k-1$ и суммируем их:
  $$
  \begin{aligned}
  \uncover<+->{ \sum\limits_{i=0}^{k-1}\left[ \varphi(x_{i+1}) - \varphi(x^*) \right] & \leq \frac{1}{2\alpha}\left[\|x_0 - x^* \|_2^2 - \|x_{k} - x^* \|_2^2\right] \\ }
  \uncover<+->{ & \leq \frac{1}{2\alpha} \|x_0 - x^* \|_2^2 }
  \end{aligned}
  $$

10. Поскольку $\varphi(x_{k})$ является убывающей последовательностью, то:
  $$
  \begin{aligned}
  \uncover<+->{ \sum\limits_{i=0}^{k-1} \varphi(x_{k})= k \varphi(x_{k}) &\leq \sum\limits_{i=0}^{k-1} \varphi(x_{i+1}) \\ }
  \uncover<+->{ \varphi(x_{k}) &\leq \frac1k \sum\limits_{i=0}^{k-1} \varphi(x_{i+1}) \\ }
  \uncover<+->{ \varphi(x_{k})  - \varphi(x^*) &\leq \frac1k \sum\limits_{i=0}^{k-1}\left[ \varphi(x_{i+1}) - \varphi(x^*)\right] \leq \frac{\|x_0 - x^* \|_2^2}{2\alpha k} }
  \end{aligned}
  $$

  . . .

  Что является стандартной оценкой $\frac{L \|x_0 - x^* \|_2^2}{2 k}$ с $\alpha = \frac1L$, или, скоростью $\mathcal{O}\left( \frac1k \right)$ для гладких выпуклых задач с градиентным спуском!



# Проксимальный градиентный метод. Сильно выпуклый случай

## Сходимость

:::{.callout-theorem}

Рассмотрим проксимальный градиентный метод
$$
x_{k+1} = \text{prox}_{\alpha r}\left(x_k - \alpha \nabla f(x_k)\right)
$$
Для критерия $\varphi(x) = f(x) + r(x)$, мы предполагаем:

::: {.nonincremental}
* $f$ является $\mu$-сильно выпуклой, дифференцируемой, $\text{dom}(f) = \mathbb{R}^n$, и $\nabla f$ является липшицевой с константой $L > 0$.
* $r$ выпукла, и $\text{prox}_{\alpha r}(x_k) = \text{arg}\min\limits_{x\in \mathbb{R}^n} \left[ \alpha r(x) +  \frac{1}{2} \|x - x_k\|^2_2 \right]$ может быть вычислен.
:::

Проксимальный градиентный спуск с фиксированным шагом $\alpha \leq 1/L$ удовлетворяет
$$
\|x_{k} - x^*\|_2^2 \leq \left(1 - \alpha \mu\right)^k \|x_{0} - x^*\|_2^2
$$
:::

Это точно соответствует скорости сходимости градиентного спуска. Обратите внимание, что исходная задача даже негладкая!

## Анализ сходимости \faGem \ \faGem[regular]

**Доказательство**

1. Учитывая расстояние до решения и используя лемму о стационарной точке:
  $$
  \begin{aligned}
  \uncover<+->{ \|x_{k+1} - x^*\|^2_2 &= \|\text{prox}_{\alpha f} (x_k - \alpha \nabla f (x_k)) - x^*\|^2_2 \\ }
  \uncover<+->{ {\scriptsize \text{лемма о стационарной точке}}  & = \|\text{prox}_{\alpha f} (x_k - \alpha \nabla f (x_k)) - \text{prox}_{\alpha f} (x^* - \alpha \nabla f (x^*)) \|^2_2 \\ }
  \uncover<+->{ {\scriptsize \text{нерастяжимость}}   & \leq \|x_k - \alpha \nabla f (x_k) - x^* + \alpha \nabla f (x^*) \|^2_2 \\ }
  \uncover<+->{ & =  \|x_k - x^*\|^2 - 2\alpha \langle \nabla f(x_k) - \nabla f(x^*), x_k - x^* \rangle + \alpha^2 \|\nabla f(x_k) - \nabla f(x^*)\|^2_2 }
  \end{aligned}
  $$

2. Теперь мы используем гладкость из анализа сходимости и сильную выпуклость: 
  $$
  \begin{aligned}
  \uncover<+->{ \text{гладкость} \;\; &\|\nabla f(x_k)-\nabla f (x^*)\|_2^2 \leq 2L\left(f(x_k)-f(x^*)-\langle\nabla f (x^*),x_k -x^*\rangle \right) \\ }
  \uncover<+->{ \text{сильная выпуклость} \;\; & - \langle \nabla f(x_k) -  \nabla f(x^*), x_k - x^* \rangle \leq - \left(f(x_k) - f(x^*) + \frac{\mu}{2} \| x_k - x^* \|^2_2 \right) - \\ - \langle \nabla f(x^*), x_k - x^* \rangle }
  \end{aligned}
  $$

## Анализ сходимости \faGem \ \faGem

3. Подставим:
  $$
  \begin{aligned}
  \uncover<+->{ \|x_{k+1} - x^*\|^2_2 &\leq \|x_k - x^*\|^2 - 2\alpha \left(f(x_k) - f(x^*) + \frac{\mu}{2} \| x_k - x^* \|^2_2 \right) - 2\alpha \langle \nabla f(x^*), x_k - x^* \rangle + \\ 
  & + \alpha^2 2L\left(f(x_k)-f(x^*)-\langle\nabla f (x^*),x_k -x^*\rangle \right)  \\ }
  \uncover<+->{ &\leq (1 - \alpha \mu)\|x_k - x^*\|^2 + 2\alpha (\alpha L - 1) \left( f(x_k) - f(x^*) - \langle \nabla f(x^*), x_k - x^* \rangle \right)}
  \end{aligned}
  $$

4. Из выпуклости $f$: $f(x_k) - f(x^*) - \langle \nabla f(x^*), x_k - x^* \rangle \geq 0$. Следовательно, если мы используем $\alpha \leq \frac1L$:
  $$
  \|x_{k+1} - x^*\|^2_2 \leq (1 - \alpha \mu)\|x_k - x^*\|^2,
  $$
  что и означает линейную сходимость метода со скоростью не хуже $1 - \frac{\mu}{L}$.

## Ускоренный проксимальный градиент ‒ *выпуклая* функция

:::{.callout-theorem}

### Ускоренный проксимальный градиентный метод

Пусть $f:\mathbb{R}^n\!\to\!\mathbb{R}$ является **выпуклой** и **$L$‑гладкой**, $r:\mathbb{R}^n\!\to\!\mathbb{R}\cup\{+\infty\}$ является правильной, замкнутой и выпуклой, $\varphi(x)=f(x)+r(x)$ имеет минимизатор $x^\star$, и предположим, что $\operatorname{prox}_{\alpha r}$ легко вычисляется для $\alpha>0$. С любым $x_0\in\operatorname{dom}r$ определим последовательность  
$$
\begin{aligned}
t_0 &= 1,\qquad y_0 = x_0,\\
x_k &= \operatorname{prox}_{\tfrac1L r}\!\bigl(y_{k-1}-\tfrac1L\nabla f(y_{k-1})\bigr),\\
t_k &= \frac{1+\sqrt{1+4t_{k-1}^2}}{2},\\
y_k &= x_k+\frac{t_{k-1}-1}{t_k}\,(x_k-x_{k-1}), \qquad k\ge 1.
\end{aligned}
$$
Для каждого $k\ge 1$
$$
\boxed{\;
\varphi(x_k)-\varphi(x^\star)\;\le\;
\frac{2L\,\|x_0-x^\star\|_2^{\,2}}{(k+1)^2}
\;}
$$ 
:::

## Ускоренный проксимальный градиент ‒ *$\mu$‑сильно выпуклая* функция

:::{.callout-theorem}

### Ускоренный проксимальный градиентный метод

Добавим, что $f$ является **$\mu$‑сильно выпуклой** ($\mu>0$).  
Установим шаг $\alpha=\tfrac1L$ и фиксированный параметр импульса  
$$
\beta\;=\;\frac{\sqrt{L/\mu}-1}{\sqrt{L/\mu}+1}.
$$
Генерируем итерации для $k\ge 0$ (возьмем $x_{-1}=x_0$):
$$
\begin{aligned}
y_k &= x_k+\beta\,(x_k-x_{k-1}),\\
x_{k+1} &= \operatorname{prox}_{\alpha r}\!\bigl(y_k-\alpha\nabla f(y_k)\bigr).
\end{aligned}
$$
Для каждого $k\ge 0$
$$
\boxed{\;
\varphi(x_k)-\varphi(x^\star)\;\le\;\Bigl(1-\sqrt{\tfrac{\mu}{L}}\Bigr)^{k} \left( \varphi(x_0) - \varphi(x^\star) + \frac{\mu}{2} \|x_0 - x^\star\|_2^2 \right)
\;}
$$
:::

# Численные эксперименты

## Квадратичный случай

$$
f(x) = \frac{1}{2m}\|Ax - b\|_2^2 + \lambda \|x\|_1 \to \min_{x \in \mathbb{R}^n}, \qquad A \in \mathbb{R}^{m \times n}, \quad \lambda\left(\tfrac{1}{m} A^TA\right) \in [\mu; L].
$$

![Гладкий выпуклый случай. Сублинейная сходимость, отсутствие сходимости в области, нет разницы между методом субградиента и проксимальным методом.](lasso_proximal_subgrad_0.pdf)

## Квадратичный случай

$$
f(x) = \frac{1}{2m}\|Ax - b\|_2^2 + \lambda \|x\|_1 \to \min_{x \in \mathbb{R}^n}, \qquad A \in \mathbb{R}^{m \times n}, \quad \lambda\left(\tfrac{1}{m} A^TA\right) \in [\mu; L].
$$

![Негладкий выпуклый случай. Сублинейная сходимость. В начале метод субградиента и проксимальный метод близки.](lasso_proximal_subgrad_1_short.pdf)

## Квадратичный случай

$$
f(x) = \frac{1}{2m}\|Ax - b\|_2^2 + \lambda \|x\|_1 \to \min_{x \in \mathbb{R}^n}, \qquad A \in \mathbb{R}^{m \times n}, \quad \lambda\left(\tfrac{1}{m} A^TA\right) \in [\mu; L].
$$

![Негладкий выпуклый случай. Если мы возьмем больше итераций, то проксимальный метод сходится с постоянным шагом, что не так для метода субградиента. Разница огромна, в то время как сложность итерации одинакова.](lasso_proximal_subgrad_1_long.pdf)

## Бинарная логистическая регрессия

$$
f(x) = \frac{1}{m}\sum_{i=1}^{m} \log(1 + \exp(-b_i(A_i x))) + \lambda \|x\|_1 \to \min_{x \in \mathbb{R}^n}, \qquad A_i \in \mathbb{R}^n, \quad b_i \in \{-1,1\}
$$

![Логистическая регрессия с $\ell_1$-регуляризацией](logistic_m_300_n_50_lambda_0.1.pdf)

## Softmax multiclass regression

![](Subgrad lr 1.00Subgrad lr 0.50Proximal lr 0.50_0.01.pdf)


## Пример: ISTA

### Iterative Shrinkage-Thresholding Algorithm (ISTA)

ISTA является популярным методом для решения задач оптимизации с $\ell_1$-регуляризацией, такой как Lasso. Он объединяет градиентный спуск с оператором сжатия для эффективного управления негладким $\ell_1$-штрафом.

* **Алгоритм**:
  - Дано $x_0$, для $k \geq 0$, повторять:
    $$
    x_{k+1} = \text{prox}_{\lambda \alpha \|\cdot\|_1} \left(x_k - \alpha \nabla f(x_k)\right),
    $$
  где $\text{prox}_{\lambda \alpha \|\cdot\|_1}(v)$ применяет оператор сжатия к каждому компоненту $v$.

* **Сходимость**:
  - Сходится со скоростью $O(1/k)$ для подходящего шага $\alpha$.

* **Применение**:
  - Эффективно для восстановления разреженных сигналов, обработки изображений и compressed sensing.

## Пример: FISTA

### Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

FISTA улучшает сходимость ISTA, включая в неё импульсное слагаемое, вдохновленное методом Нестерова.

* **Алгоритм**:
  - Инициализируем $x_0 = y_0$, $t_0 = 1$.
  - Для $k \geq 1$, обновляем:
    $$
    \begin{aligned}
    x_{k} &= \text{prox}_{\lambda \alpha \|\cdot\|_1} \left(y_{k-1} - \alpha \nabla f(y_{k-1})\right), \\
    t_{k} &= \frac{1 + \sqrt{1 + 4t_{k-1}^2}}{2}, \\
    y_{k} &= x_{k} + \frac{t_{k-1} - 1}{t_{k}}(x_{k} - x_{k-1}).
    \end{aligned}
    $$
  
* **Сходимость**:
  - Улучшает скорость сходимости до $O(1/k^2)$.
  
* **Применение**:
  - Особенно полезен для больших задач в машинном обучении и обработке сигналов, где $\ell_1$-штраф индуцирует разреженность.

## Пример: задача восстановления матрицы (Matrix Completion)

### Решение задачи Matrix Completion

Задачи matrix completion стремятся заполнить пропущенные элементы частично наблюдаемой матрицы при определенных предположениях, обычно низкого ранга. Это может быть сформулировано в виде задачи минимизации, включающую ядерную норму (сумму сингулярных значений), которая продвигает решения низкого ранга.

* **Формулировка задачи**:
  $$
  \min_{X} \frac{1}{2} \|P_{\Omega}(X) - P_{\Omega}(M)\|_F^2 + \lambda \|X\|_*,
  $$
  где $P_{\Omega}$ проецирует на наблюдаемое множество $\Omega$, и $\|\cdot\|_*$ обозначает ядерную норму.

* **Проксимальный оператор**:
  - Проксимальный оператор для ядерной нормы включает сингулярное разложение (SVD) и сжатие сингулярных значений.
  
* **Алгоритм**:
  - Можно применять аналогичные проксимальные методы или ускоренные проксимальные методы; основной вычислительный расход приходится на выполнение SVD.

* **Применение**:
  - Широко используется в рекомендательных системах, восстановлении изображений и других областях, где данные естественно представлены в виде матриц, но частично наблюдаемы.

## Summary

* Если использовать структуру задачи, можно превзойти нижние оценки для неструктурированной постановки.
* Проксимальный метод для задачи с $L$-гладкой выпуклой функцией $f$ и выпуклой функцией $r$ с вычислимым проксимальным оператором имеет ту же скорость сходимости, что и метод градиентного спуска для $f$. Свойства гладкости/негладкости $r$ на сходимость не влияют.
* Кажется, что если $f = 0$, то любая негладкая задача может быть решена таким методом. Вопрос: это правда? 
    
    . . .

   Если разрешить численно неточный проксимальный оператор, то действительно можно решать любую негладкую задачу оптимизации. Но с теоретической точки зрения это не лучше субградиентного спуска, поскольку для решения проксимальной подзадачи используется вспомогательный метод (например, тот же субградиентный спуск).
* Проксимальный метод является общим современным фреймворком для многих численных методов. Далее развиваются ускоренные, стохастические, приближенные двойственные методы и т.д.
* Дополнительные материалы: разбиение по проксимальному оператору, схема Дугласа—Рачфорда, задача наилучшего приближения, разбиение на три оператора.