---
title: Analyse des données - 1 Introduction
author: Stone SHI
date: 2021-10-18 3:40:00 +0200
categories: [Blogging, Study]
tags: [Data_analysis]
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# I. Professeur

Philippe MOUTTOU

**Boite:** [THALES](https://www.thalesgroup.com/en)

**Post:** Recherches de [TRL 4](#TRL)

**Dommaine:** Analyse des données

**Email:** philippe.mouttou@thalesgroup.com

# II. Contexte

poser le pb 

1. Definition du pb

2. System d'information

que'est-ce qu'on observe.

3. comment - méthodes

4. Deux monde - exhaustif et échantillon

Exhaustivité - j'ai tous les cas dissponibles - Big Data

Échantillon - Probabilité - 

5. Statistique descriptive

moyenne, éout tupe, variance, fréquence

indicateur de causalité

6. ML et AD

Factorisation et classification

# III. Évaluation

groupe de 2 ou 3, avec 4 transparents

1. Nature de pb.

2. Ce qu je fait avec ML/AD

Méthode de factorielle.

3. Qu'est-ce qui ne marche pas.

4. Qu'est-ce qu'il faudrai faire.

# IV. Cours 1

## 1. Déscription

$$x(x^1, x^2, ... ,x^p)$$

## 2. Illustration

Notation $\mathbb{R}$

Espace vectoriel euclidien

Espace affine euclidien

$$ x=\left[ \begin{matrix} x^1 \\ x^2 \\ \vdots \\ x^p \end{matrix} \right], \quad y=\left[ \begin{matrix} y^1 \\ y^2 \\ \vdots \\ y^p \end{matrix} \right] $$

On définit,

$$ (x|y) = \displaystyle \sum ^{p}_{i = 1}{x^iy^i} $$

$$ \left \| x \right \| = \sqrt {(x|x)} $$

$$ d^2(x, y) = \displaystyle \sum ^{p}_{i = 1}{(x^i-y^i)^2} $$

## 3. Problème

Soit $n \in \mathbb{N}$, soit un ensmeble de $n$ points 

$$ N=(x_i)_{1\leq i \leq n} \in (\mathbb{R}^p)^n $$

avec les poids 

$$ P=(p_i)_{1 \leq i \leq n} \in {[0,+ \infty [}^n $$ 

tels que $ \displaystyle \sum _{i}{p_i}=1 $

On définit $g$ le `centre de grativité` sous cette condition 

$$ g = \displaystyle \sum^{n}_{i=1}{p_ix_i} $$

le Critère de Fidélité:

$$\tag{Fidelity criteria} \operatorname {Max}(\displaystyle \sum _{i}{\displaystyle \sum _{j}{p_ip_jd^2(\hat{x_i}, \hat{x_j})}})$$

Definisson $ \operatorname {M^t} (H) = \displaystyle \sum _{i}{p_i {\left \| x_i - \hat{x_i} \right \|}^2}$

le Critère de Proximité:

$$ \tag{Proximity Criteria} \operatorname {Min} (\operatorname {M^t} (H) ) $$

## 4. Résultats

## 5. Résolution

## 6. Critère


# V. Infos Complémentaires

<div id="TRL"></div>
## Technical Reacliness Level (TRL) 

1 - 9 

1 - 3 Recherche 4 - 6 Recherche appliquée
