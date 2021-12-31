---
title: 区块链 - 区块链技术习题
author: Stone SHI
date: 2021-12-16 6:07:00 +0200
categories: [Blogging, Block chain]
tags: [Block chain]
---

# 参考资料

[blockchain.com](https://www.blockchain.com/)

[btc.com](https://btc.com/)

[myetherwallet.com](https://www.myetherwallet.com/)

[大白话科普区块链原理【子豪兄区块链讲义】](https://zhuanlan.zhihu.com/p/50767485)

# 习题

## 1 - MD5 "BLOCKCHAIN"
Q:

- Quel est le MD5 de la chaîne de 10 caractères en majuscule BLOCKCHAIN ?
- Combien de bits (binary digits) fait ce hash ?

A:

MD5 de "BLOCKCHAIN": e4b25d8c911628d527b2c97039a9f593
nombre de bits: 128

## 2 - SHA3 "BLOCKCHAIN"
Q:

Quel est le SHA3-256 la chaîne de 10 caractères en majuscule `BLOCKCHAIN` ?

A:

bd6af45b03ce764acba7f330ddb9383325f2ce48e0d1708207c2ea01c7855b97

## 3 - Collision SHA3
Q:

Combien d'opérations sont en moyenne nécessaires pour trouver une collision sur SHA3-256 ?

A:

3.4028237e+38

## 4 - H99 - hashage de 5432
Q:

Soit la fonction de hashage d'entiers H99(i) définie comme i modulo 99. Ses qualités cryptographiques sautent aux yeux.

- Que vaut H99(5432) ?

A:

86

## 5 - H99 - inversion
Q:

Quel est le premier entier supérieur à 1000 dont le H99 vaut 0 ?

A:

1089

## 6 - H99 - arbre de Merkle
Q:

Quelle est la racine de l'arbre binaire de Merkle calculé sur les feuilles 1234 5678 9876 5430 avec la fonction H99 ?

On aggrégera les hashs à chaque niveau en concaténant simplement les représentations décimales. Il y a 7 hashs à calculer au total.

À quoi peut faire penser ce résultat ? 😉

A:

4635 7584 8160 42

La grande question sur la vie, l'univers et le reste

## 7 - Hash - preuve de Merkle
Q:

Soit la fonction de hash h, les éléments e0 à e3 dont les hashs sont f0 = h(e0) à f3 = h(e3), et les hashs de l'arbre binaire de Merkle r1 = h(f0 . f1), r2 = h(f2 . f3), r0 = h(r1 . r2) (. est la concaténation).

Quelle formule démontre que l'élément e2 appartient à l'arbre de Merkle de racine r0 ?

A:

f2 = h(e2)

r2 = h(f2.f3)

r0 = h(r1.r2)

## 8 - BTC - fonction de hash
Q:

Quelle fonction de hashage est utilisée par bitcoin ?

A:

SHA-256

## 9 - BTC - premier block
Q:

Quel est le numéro du premier bloc bitcoin ?

A:

0

## 10 - BTC - bloc précédent
Q:

Quel est le hash du bloc précédent du premier bloc (bloc qui n'existe pas, bien sûr) ?

A:

0

## 11 - BTC - date du premier bloc
Q:

À quelle date ce bloc a-t-il été généré ?

A:

3 janvier 2009

## 12 - BTC - rémunération du premier bloc
Q:

Combien de bitcoins ont été créés par le minage du premier bloc ?

A:

50 btc

## 13 - BTC - bloc 210000
Q:

Quel est la particularité du bloc numéro 210000 ?

A:

25 BTC de rémunération

## 14 - BTC - rémunération 12.5
Q:

À quel bloc la rémunération des mineurs est-elle passée à 12.5 BTC ?

A:

420000

## 15 - BTC - rémunération 6.25
Q:

À quel bloc la rémunération des mineurs de bitcoins est-elle passée à 6.25 ETC ?

A:

630000

## 16 - BTC - minage de bloc 561744
Q:

Combien de temps a duré le minage du block 561744 ?

A:

1 second

## 16 - BTC - minage de bloc 561744
Q:

Combien de temps a duré le minage du block 561744 ?

A:

1 second

## 17 - BTC - minage du bloc 711651
Q:

Combien de temps a duré le minage du block 711651 ?

A:

58 minutes 32 seconds

## 18 - BTC - bloc 666666
Q:

Regarder le bloc bitcoin 666666.

- de quand date-t-il ?
- combien de transactions contient-il ?
- quel est le montant des bitcoins échangés ?

A:

January 19, 2021
2728
880.73126915 BTC

## 19 - ETH - rémunération initiale des mineurs
Q:

Quelle est la rémunération statique du 1er bloc de la blockchain Ethereum ?

A:

5 ETH

## 20 - ETH - nouvelle rémunération
Q:

À quel bloc la rémunération des mineurs passe-t-elle à 3 ETH ?

A:

4370000

## 22 - ETH - prénom hexadécimal débile (PHD)
Q:

Traduisez votre prénom en 4 chiffres hexadécimaux de la façon suivante :

- Les lettres de A à F sont gardées, puis
- substituer les autres lettres par des chiffres : 1 pour I J L Y, 2 pour R Z, 3 pour X, 4 pour H K, 5 pour S, 6 pour G, 7 pour P T, 8 pour M N, 9 pour U V W et 0 pour O Q.
- tronquer après 4 caractères

Calvin est ainsi traduit en "CA1918", tronqué à "CA19", Hobbes est "40BBE5" tronqué à "40BB".

Quel est votre PHD ?

A:

5493

## 23 - ETH - création d'un portefeuille vaniteux
Q:

- Connectez-vous à Vanity Eth pour vous créer un portefeuille Ethereum dont l'adresse commence justement par votre PHD.

- Choisissez-vous un mot de passe super top secret.

- Téléchargez et sauvegardez le fichier résultat, et collez-en le contenu ci-après.

A:

{"address":"0x549399259eacf2545f98622ace39e369e990dbb2","crypto":{"kdf":"pbkdf2","kdfparams":{"c":262144,"dklen":32,"prf":"hmac-sha256","salt":"a30aa67e621715c34faeb5b8cacc61e9cf3f455c53b4e932c260f4c2316572ea"},"cipher":"aes-128-ctr","ciphertext":"6364e71d1bdba2b78651313251a4417f17fb4c5974c92acbbc0f943282849f86","cipherparams":{"iv":"36fcfa4d4e1e09846afeae353e894b91"},"mac":"bf32e0202109fa6b15c8e9c952e5967155403f56f41771148b881dca7bbc079b"},"id":"8c8b7326-cf4e-4d22-a764-da8868bb59c2","version":3}

## 24 - ETH - examen du portefeuille (1)
Q:

Quel est le format du fichier ?

A:

json file

## 25 - ETH - examen du portefeuille (2)
Q:

Quel algorithme de chiffrement est utilisé pour protéger la clef privée ?

A:

AES

## 26 - ETH - chargement dans MyEtherWallet
Q:

Aller sur MyEtherWallet pour visualiser votre portefeuille.

- Quelle est votre adresse Ethereum ?

A:

0x549399259eacf2545F98622acE39e369e990Dbb2

## 27 - ETH - nombre d'octets de l'adresse
Q:

Combien d'octets fait-elle ?

A:

20

## 28 - ETH - contenu de l'adresse
Q:

Pourquoi contient-elle des minuscules et des majuscules ?

A:

Because an Ethereum address is a 42-character hexadecimal address derived from the last 20 bytes of the public key controlling the account with 0x appended in front.

## 29 - ETH - clef privée
Q:

Quelle est la taille de la clef privée en chiffres binaires ?

A:

256 bits

## 30 - ETH - solde (1)
Q:

Quel est le solde de votre compte ?

A:

0.00 ETH

## 31 - ETH - solde (2)
Q:

Quel était le solde du compte Ethereum 0xAceBabe64807cb045505b268ef253D8fC2FeF5Bc au 1er janvier 2019 ?

A:

0.2 ETH

## 32 - ETH - solde (3)
Q:

Quel était le solde du compte Ethereum 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2 au 30 novembre 2021 ?

A:

交易频率太高导致手动难以推算之前的账户余额。

## 33 - ETH - Ethermine (1)
Q:

- Quelle est l'adresse du portefeuille du groupe de mineurs (mining pool) Ethermine ?
- Depuis quand ce portefeuille est-il utilisé ?

A:

0xEA674fdDe714fd979de3EdF0F56AA9716B898ec8
2016/3/21

## 34 - ETH - Ethermine (2)
Q:

À l'instant où vous répondez à cette question :

- quel est le numéro du dernier bloc qui contient une transaction impliquant ce compte ?

A:

13783867

## 35 - ETH - Wrapped Ether 0xc02a...
Q:

- Quel est le solde du compte Wrapped Ether 0xc02a... en ETH ?
- Quelle est la valeur équivalente (on arrondira au milliard de dollars le plus proche:-)

A:

7318110.01536ETH
30 milliard de dollars