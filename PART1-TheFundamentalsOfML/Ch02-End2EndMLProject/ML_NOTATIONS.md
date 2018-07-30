<h1 align="center">Notations</h1>

> **Before Starting**: If you have a need of notations always, keep this markdown open in a separate tab.


* ![m](./images/m.png) is the number of instances in the dataset you are measuring the RMSE on.
  * For example, if you have data for 2000 districts then, *m* = 2000.
* ![xi](./images/xi.png) is a vector of all the feature values of the ![ith](./images/ith.png) instance in the dataset and ![xi](./images/yi.png) is its label.
  * For example, if the first district in dataset is located at longitude -118.29 degree, lattitude 33.91 degree, and it has 1,416 inhabitants with a median income of \$38,372, and the median house value is \$156,400 then:

![x1](./images/x1.png)

and:

![y1](./images/y1.png)

* **X** is a matrix containing all the feature values (excluding labels) of all instances in the dataset. There is one row per instance and ![ith](./images/ith.png) row is equal to transpose of ![xi](./images/xi.png), noted ![xi](./images/xiT.png)
  * Example **X** matrix:

![X](./images/X.png)

* ![h](./images/h.png) is your prediction function, a.k.a. *Hypothesis*. To predict value for instance ![xi](./images/xi.png) we use - ![yhati](./images/yhati.png).

* *RMSE*(**X**, *h*) and *MAE*(**X**, *h*) are the cost functions.
* *RMSE* corresponds to *Eucledian norm*. It is also called the ![l2](./images/l2.png) *norm*, noted ![l2_notation1](./images/l2_noted1.png) or ![l2_notation2](./images/l2_noted2.png).
* *MAE* corresponds to ![l1](./images/l1.png) norm, noted ![l1_notation](./images/l1_noted.png). It is sometimes called *Manhatton norm* as it measure distance between two points in a city if you can only travel along orthogonal city blocks.
* k-norm of a vector **v** is given by, ![lk_norm](./images/lk_norm.png). 
* Higher the value of *k* in k norm the higher the more it focuses on larger values and forget smaller ones.
